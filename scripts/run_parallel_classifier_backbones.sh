#!/usr/bin/env bash
# Parallel timm classifier backbone sweep (ConvNeXt, EfficientNet, ResNet, Swin, ViT, ...).
# Uses one base YAML + --classifier-model per job (SwinV2 @256 uses a separate base config).
# Sequential classifier configs support:
#   - both phases (default): synth pretrain -> real finetune
#   - synth only: CLASSIFIER_END_PHASE=1
#   - real only:  CLASSIFIER_START_PHASE=1  (expects phase-0 checkpoint on disk)
#
# Logging:
#   - by default, jobs log to logs/classifier_backbone_sweep_<mode>_<timestamp>/
#   - set CLASSIFIER_SWEEP_LOG_DIR=/path to choose a directory
#   - set CLASSIFIER_SWEEP_LOG_DIR=none (or -) to keep child stdout/stderr in the terminal
#
# Usage:
#   bash scripts/run_parallel_classifier_backbones.sh
#
#   CLASSIFIER_NUM_GPUS=4 \
#   CLASSIFIER_SWEEP_LOG_DIR="logs/classifier_backbone_sweep" \
#   bash scripts/run_parallel_classifier_backbones.sh
#
# GPU selection:
#   CLASSIFIER_GPUS="0 1 2 3"  # explicit list, highest priority
#   CLASSIFIER_NUM_GPUS=4      # first N GPUs from nvidia-smi
#   unset both                 # auto-detect all GPUs via nvidia-smi; fallback: 0..7
#
# Sequential phase control:
#   CLASSIFIER_END_PHASE=1 bash scripts/run_parallel_classifier_backbones.sh
#   CLASSIFIER_START_PHASE=1 bash scripts/run_parallel_classifier_backbones.sh
#   CLASSIFIER_INIT_WEIGHTS=/abs/path/best.pt CLASSIFIER_START_PHASE=1 bash scripts/run_parallel_classifier_backbones.sh
#   CLASSIFIER_RESUME_WEIGHTS=last CLASSIFIER_START_PHASE=1 bash scripts/run_parallel_classifier_backbones.sh
#   CLASSIFIER_PHASE1_EPOCHS=60 bash scripts/run_parallel_classifier_backbones.sh
# General regime override:
#   CLASSIFIER_REGIME=real_only bash scripts/run_parallel_classifier_backbones.sh
#   CLASSIFIER_EXCLUDE_MODELS="model_convnext_small model_convnext_tiny" bash scripts/run_parallel_classifier_backbones.sh
#   CLASSIFIER_MODELS="model_convnext_small model_convnext_tiny" bash scripts/run_parallel_classifier_backbones.sh
#
# Optional: CLASSIFIER_BASE_CONFIG=..., CLASSIFIER_SWINV2_BASE_CONFIG=...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

BASE="${CLASSIFIER_BASE_CONFIG:-${ROOT}/configs/experiments/sweep_classifier_stageb_base.yaml}"
BASE_SWINV2="${CLASSIFIER_SWINV2_BASE_CONFIG:-${ROOT}/configs/experiments/sweep_classifier_swinv2_256.yaml}"

# Families aligned with common domain-generalization / backbone tables (ImageNet-scale pretrain via timm).
MODELS=(
  convnext_tiny
  convnext_small
  convnext_base
  convnext_large
  efficientnet_b0
  efficientnet_b1
  efficientnet_b2
  tf_efficientnet_b3_ns
  resnet34
  resnet50
  resnet101
  swin_tiny_patch4_window7_224
  swin_small_patch4_window7_224
  swin_base_patch4_window7_224
  vit_small_patch16_224
  vit_base_patch16_224
)

SWINV2_MODELS=(
  swinv2_tiny_window8_256
  swinv2_small_window8_256
  # timm has no swinv2_base_window12_256 entry. This is its 256px
  # ImageNet-22k -> ImageNet-1k fine-tuned counterpart.
  swinv2_base_window12to16_192to256
)

_read_env_list() {
  local raw="${1:-}"
  raw="${raw//,/ }"
  read -r -a _ENV_LIST <<< "${raw}"
  printf '%s\n' "${_ENV_LIST[@]}"
}

_detect_gpus() {
  local -a detected=()
  local gpu
  if command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r gpu; do
      gpu="${gpu//[[:space:]]/}"
      [[ -n "${gpu}" ]] && detected+=("${gpu}")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
  fi

  if [[ "${#detected[@]}" -eq 0 ]]; then
    detected=(0 1 2 3 4 5 6 7)
  fi

  printf '%s\n' "${detected[@]}"
}

if [[ -n "${CLASSIFIER_GPUS:-}" ]]; then
  read -r -a GPUS <<< "${CLASSIFIER_GPUS}"
else
  GPUS=()
  while IFS= read -r gpu; do
    [[ -n "${gpu}" ]] && GPUS+=("${gpu}")
  done < <(_detect_gpus)

  if [[ -n "${CLASSIFIER_NUM_GPUS:-}" ]]; then
    if ! [[ "${CLASSIFIER_NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${CLASSIFIER_NUM_GPUS}" -lt 1 ]]; then
      echo "Invalid CLASSIFIER_NUM_GPUS=${CLASSIFIER_NUM_GPUS} (must be a positive integer)." >&2
      exit 1
    fi
    if [[ "${CLASSIFIER_NUM_GPUS}" -gt "${#GPUS[@]}" ]]; then
      echo "[classifier][warn] CLASSIFIER_NUM_GPUS=${CLASSIFIER_NUM_GPUS}, but only detected ${#GPUS[@]} GPU(s); using ${#GPUS[@]}." >&2
    fi
    GPUS=( "${GPUS[@]:0:${CLASSIFIER_NUM_GPUS}}" )
  fi
fi
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPU ids available (CLASSIFIER_GPUS is empty / auto-detect failed)." >&2
  exit 1
fi

CLASSIFIER_INCLUDE_MODELS=()
if [[ -n "${CLASSIFIER_MODELS:-}" ]]; then
  while IFS= read -r item; do
    [[ -n "${item}" ]] && CLASSIFIER_INCLUDE_MODELS+=("${item}")
  done < <(_read_env_list "${CLASSIFIER_MODELS}")
fi

CLASSIFIER_EXCLUDE_MODELS_LIST=()
if [[ -n "${CLASSIFIER_EXCLUDE_MODELS:-}" ]]; then
  while IFS= read -r item; do
    [[ -n "${item}" ]] && CLASSIFIER_EXCLUDE_MODELS_LIST+=("${item}")
  done < <(_read_env_list "${CLASSIFIER_EXCLUDE_MODELS}")
fi

_classifier_phase_args_suffix() {
  local suffix=""
  if [[ -n "${CLASSIFIER_REGIME:-}" ]]; then
    suffix+=" --regime $(printf '%q' "${CLASSIFIER_REGIME}")"
  fi
  if [[ -n "${CLASSIFIER_START_PHASE:-}" ]]; then
    suffix+=" --classifier-start-phase $(printf '%q' "${CLASSIFIER_START_PHASE}")"
  fi
  if [[ -n "${CLASSIFIER_END_PHASE:-}" ]]; then
    suffix+=" --classifier-end-phase $(printf '%q' "${CLASSIFIER_END_PHASE}")"
  fi
  if [[ -n "${CLASSIFIER_PHASE1_EPOCHS:-}" ]]; then
    suffix+=" --classifier-phase1-epochs $(printf '%q' "${CLASSIFIER_PHASE1_EPOCHS}")"
  fi
  if [[ -n "${CLASSIFIER_INIT_WEIGHTS:-}" ]]; then
    suffix+=" --classifier-init-weights $(printf '%q' "${CLASSIFIER_INIT_WEIGHTS}")"
  fi
  if [[ -n "${CLASSIFIER_RESUME_WEIGHTS:-}" ]]; then
    suffix+=" --classifier-resume-weights $(printf '%q' "${CLASSIFIER_RESUME_WEIGHTS}")"
  fi
  printf '%s' "${suffix}"
}

_classifier_model_matches_token() {
  local kind="$1"
  local model="$2"
  local token="$3"

  [[ -z "${token}" ]] && return 1

  case "${kind}" in
    model)
      [[ "${token}" == "${model}" ]] \
        || [[ "${token}" == "model:${model}" ]] \
        || [[ "${token}" == "model_${model}" ]]
      ;;
    swinv2)
      [[ "${token}" == "${model}" ]] \
        || [[ "${token}" == "swinv2:${model}" ]] \
        || [[ "${token}" == "swinv2_${model}" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

_classifier_model_selected() {
  local kind="$1"
  local model="$2"
  local token

  if [[ "${#CLASSIFIER_INCLUDE_MODELS[@]}" -gt 0 ]]; then
    local matched=0
    for token in "${CLASSIFIER_INCLUDE_MODELS[@]}"; do
      if _classifier_model_matches_token "${kind}" "${model}" "${token}"; then
        matched=1
        break
      fi
    done
    if [[ "${matched}" -ne 1 ]]; then
      return 1
    fi
  fi

  for token in "${CLASSIFIER_EXCLUDE_MODELS_LIST[@]}"; do
    if _classifier_model_matches_token "${kind}" "${model}" "${token}"; then
      return 1
    fi
  done

  return 0
}

_yaml_scalar_value() {
  local path="$1"
  local key="$2"
  awk -F': *' -v k="${key}" '$1 == k { print $2; exit }' "${path}"
}

_classifier_has_resume_ckpt() {
  local cfg="$1"
  local model="$2"
  local resume_mode="${CLASSIFIER_RESUME_WEIGHTS:-auto}"

  if [[ -n "${CLASSIFIER_INIT_WEIGHTS:-}" ]]; then
    [[ -f "${CLASSIFIER_INIT_WEIGHTS}" ]]
    return
  fi

  local exp_name output_root out_abs run_dir suffix t
  exp_name="$(_yaml_scalar_value "${cfg}" "experiment_name")"
  output_root="$(_yaml_scalar_value "${cfg}" "output_root")"
  if [[ -z "${exp_name}" || -z "${output_root}" ]]; then
    return 1
  fi
  if [[ "${output_root}" = /* ]]; then
    out_abs="${output_root}"
  else
    out_abs="${ROOT}/${output_root}"
  fi
  # Mirror paleo_ocr/experiments/run_experiment.py:_classifier_phase1_epoch_slug so the
  # bash pre-check looks in the same suffixed dir Python uses (e.g. cls_sweep_stagea_base__convnext_tiny__real60e).
  suffix=""
  if [[ -n "${CLASSIFIER_PHASE1_EPOCHS:-}" ]]; then
    t="${CLASSIFIER_PHASE1_EPOCHS,,}"
    if [[ ! "${t}" =~ ^(yaml|default|file)$ ]] && [[ "${t}" =~ ^[0-9]+$ ]] && (( t > 0 )); then
      suffix="__real${t}e"
    fi
  fi
  run_dir="${out_abs}/${exp_name}__${model}${suffix}/classifier_run"

  # Phase-0 dir is named after the first phase's `name` (e.g. pretrain_stagea vs pretrain_synth).
  # Check both stagea and synth candidates, plus the legacy flat classifier_run/{best,last}.pt.
  local ph0_stagea="phase_00_pretrain_stagea"
  local ph0_synth="phase_00_pretrain_synth"
  case "${resume_mode}" in
    best)
      [[ -f "${run_dir}/${ph0_stagea}/best.pt" ]] \
        || [[ -f "${run_dir}/${ph0_synth}/best.pt" ]] \
        || [[ -f "${run_dir}/best.pt" ]]
      ;;
    last)
      [[ -f "${run_dir}/${ph0_stagea}/last.pt" ]] \
        || [[ -f "${run_dir}/${ph0_synth}/last.pt" ]] \
        || [[ -f "${run_dir}/last.pt" ]]
      ;;
    auto)
      [[ -f "${run_dir}/${ph0_stagea}/best.pt" ]] \
        || [[ -f "${run_dir}/${ph0_stagea}/last.pt" ]] \
        || [[ -f "${run_dir}/${ph0_synth}/best.pt" ]] \
        || [[ -f "${run_dir}/${ph0_synth}/last.pt" ]] \
        || [[ -f "${run_dir}/best.pt" ]] \
        || [[ -f "${run_dir}/last.pt" ]]
      ;;
    *)
      echo "Invalid CLASSIFIER_RESUME_WEIGHTS=${resume_mode} (use auto, best, or last)" >&2
      return 1
      ;;
  esac
}

_classifier_phase_label() {
  local start="${CLASSIFIER_START_PHASE:-0}"
  local end="${CLASSIFIER_END_PHASE:-}"
  if [[ -n "${end}" ]]; then
    if [[ "${start}" == "0" && "${end}" == "1" ]]; then
      printf '%s' "phase0"
      return
    fi
    printf 'phase%s_to_%s' "${start}" "$((end - 1))"
    return
  fi
  if [[ -n "${CLASSIFIER_START_PHASE:-}" && "${CLASSIFIER_START_PHASE}" != "0" ]]; then
    printf 'phase%s' "${CLASSIFIER_START_PHASE}"
    return
  fi
  printf '%s' "full"
}

LOGGING_ENABLED=1
if [[ -n "${CLASSIFIER_SWEEP_LOG_DIR:-}" ]]; then
  if [[ "${CLASSIFIER_SWEEP_LOG_DIR}" == "none" || "${CLASSIFIER_SWEEP_LOG_DIR}" == "-" ]]; then
    LOGGING_ENABLED=0
  fi
else
  CLASSIFIER_SWEEP_LOG_DIR="${ROOT}/logs/classifier_backbone_sweep_$(_classifier_phase_label)_$(date +%Y%m%d_%H%M%S)"
fi

if [[ "${LOGGING_ENABLED}" -eq 1 ]]; then
  mkdir -p "${CLASSIFIER_SWEEP_LOG_DIR}"
  : > "${CLASSIFIER_SWEEP_LOG_DIR}/jobs.tsv"
fi

FILTERED_MODELS=()
for m in "${MODELS[@]}"; do
  if _classifier_model_selected "model" "${m}"; then
    FILTERED_MODELS+=("${m}")
  fi
done
MODELS=("${FILTERED_MODELS[@]}")

FILTERED_SWINV2_MODELS=()
for m in "${SWINV2_MODELS[@]}"; do
  if _classifier_model_selected "swinv2" "${m}"; then
    FILTERED_SWINV2_MODELS+=("${m}")
  fi
done
SWINV2_MODELS=("${FILTERED_SWINV2_MODELS[@]}")

total_jobs=$((${#MODELS[@]} + ${#SWINV2_MODELS[@]}))
# Concurrency cap (RAM/FUSE control). Default: one job per GPU (queued).
# The classifier grid has 16-18 models; launching all at once on 6 GPUs triples RAM
# and FUSE load. Set CLASSIFIER_MAX_PARALLEL=0 for the old "all at once" behavior.
MAX_PARALLEL="${CLASSIFIER_MAX_PARALLEL:-${#GPUS[@]}}"
echo "[classifier] base (224): ${BASE}"
echo "[classifier] base (SwinV2 @256): ${BASE_SWINV2}"
echo "[classifier] jobs: ${total_jobs} (${#MODELS[@]} @224 + ${#SWINV2_MODELS[@]} @256), gpus: ${GPUS[*]}"
echo "[classifier] max parallel: ${MAX_PARALLEL} (CLASSIFIER_MAX_PARALLEL; 0 = all at once)"
if [[ -n "${CLASSIFIER_START_PHASE:-}" || -n "${CLASSIFIER_END_PHASE:-}" ]]; then
  echo "[classifier] phase args:${CLASSIFIER_START_PHASE:+ start=${CLASSIFIER_START_PHASE}}${CLASSIFIER_END_PHASE:+ end=${CLASSIFIER_END_PHASE}}"
fi
if [[ -n "${CLASSIFIER_PHASE1_EPOCHS:-}" ]]; then
  echo "[classifier] phase-1 finetune epochs override: ${CLASSIFIER_PHASE1_EPOCHS}"
fi
if [[ -n "${CLASSIFIER_RESUME_WEIGHTS:-}" ]]; then
  echo "[classifier] resume weights: ${CLASSIFIER_RESUME_WEIGHTS}"
fi
if [[ -n "${CLASSIFIER_REGIME:-}" ]]; then
  echo "[classifier] regime: ${CLASSIFIER_REGIME}"
fi
if [[ "${#CLASSIFIER_INCLUDE_MODELS[@]}" -gt 0 ]]; then
  echo "[classifier] include models: ${CLASSIFIER_INCLUDE_MODELS[*]}"
fi
if [[ "${#CLASSIFIER_EXCLUDE_MODELS_LIST[@]}" -gt 0 ]]; then
  echo "[classifier] exclude models: ${CLASSIFIER_EXCLUDE_MODELS_LIST[*]}"
fi
if [[ "${LOGGING_ENABLED}" -eq 1 ]]; then
  echo "[classifier] log dir: ${CLASSIFIER_SWEEP_LOG_DIR}"
else
  echo "[classifier] log dir: disabled (child output goes to terminal)"
fi

pids=()
job_descs=()
job_idx=0
num_gpus="${#GPUS[@]}"
skipped=0

# Wait until fewer than MAX_PARALLEL child jobs are still running.
_wait_for_slot() {
  [[ "${MAX_PARALLEL}" -gt 0 ]] || return 0
  while true; do
    local alive=0
    for p in "${pids[@]:-}"; do
      if [[ -n "${p}" ]] && kill -0 "${p}" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    (( alive < MAX_PARALLEL )) && return
    # Reap one finished child to free a slot (also clears zombies).
    wait -n 2>/dev/null || true
  done
}

launch_job() {
  local gpu="$1"
  local kind="$2"
  local target="$3"
  local cmd="$4"
  local desc="${kind}:${target}@gpu${gpu}"

  echo "[classifier] GPU ${gpu} <- ${kind}:${target}"
  if [[ "${LOGGING_ENABLED}" -eq 1 ]]; then
    local safe_target
    safe_target="$(echo "${target}" | tr '/ ' '__')"
    local log_file="${CLASSIFIER_SWEEP_LOG_DIR}/$(printf '%03d' "${job_idx}")_${kind}_${safe_target}.log"
    (
      CUDA_VISIBLE_DEVICES="${gpu}" bash -c "${cmd}"
    ) >"${log_file}" 2>&1 &
    echo -e "$(printf '%03d' "${job_idx}")\t${desc}\t${log_file}" >> "${CLASSIFIER_SWEEP_LOG_DIR}/jobs.tsv"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" bash -c "${cmd}" &
  fi

  pids+=($!)
  job_descs+=("${desc}")
  job_idx=$((job_idx + 1))
}

for m in "${MODELS[@]}"; do
  if [[ -n "${CLASSIFIER_START_PHASE:-}" && "${CLASSIFIER_START_PHASE}" != "0" ]]; then
    if ! _classifier_has_resume_ckpt "${BASE}" "${m}"; then
      echo "[classifier][skip] model:${m} (missing phase-0 checkpoint for resume mode ${CLASSIFIER_RESUME_WEIGHTS:-auto})"
      skipped=$((skipped + 1))
      continue
    fi
  fi
  _wait_for_slot
  gpu="${GPUS[$((job_idx % num_gpus))]}"
  launch_job \
    "${gpu}" \
    "model" \
    "${m}" \
    "python -m paleo_ocr.experiments.run_experiment --config $(printf '%q' "${BASE}") --classifier-model $(printf '%q' "${m}")$(_classifier_phase_args_suffix)"
done

for m in "${SWINV2_MODELS[@]}"; do
  if [[ -n "${CLASSIFIER_START_PHASE:-}" && "${CLASSIFIER_START_PHASE}" != "0" ]]; then
    if ! _classifier_has_resume_ckpt "${BASE_SWINV2}" "${m}"; then
      echo "[classifier][skip] swinv2:${m} (missing phase-0 checkpoint for resume mode ${CLASSIFIER_RESUME_WEIGHTS:-auto})"
      skipped=$((skipped + 1))
      continue
    fi
  fi
  _wait_for_slot
  gpu="${GPUS[$((job_idx % num_gpus))]}"
  launch_job \
    "${gpu}" \
    "swinv2" \
    "${m}" \
    "python -m paleo_ocr.experiments.run_experiment --config $(printf '%q' "${BASE_SWINV2}") --classifier-model $(printf '%q' "${m}")$(_classifier_phase_args_suffix)"
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No classifier jobs were launched." >&2
  exit 1
fi

if [[ "${skipped}" -gt 0 ]]; then
  echo "[classifier] skipped jobs: ${skipped}"
fi

failures=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if ! wait "${pid}"; then
    echo "[classifier][fail] ${job_descs[$i]}" >&2
    failures=$((failures + 1))
  fi
done

if [[ "${failures}" -gt 0 ]]; then
  echo "ERROR: ${failures} classifier job(s) failed (out of ${#pids[@]})." >&2
  exit 1
fi

echo "Classifier backbone sweep finished (${#pids[@]} jobs)."
