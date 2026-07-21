import os
import json
import sys
import threading
import webbrowser
import time
from pathlib import Path
from flask import Flask, jsonify, request, send_file, Response

# ===========================
# CONFIGURATION
# ===========================
# Путь к рабочей директории (где лежат images и manifest.jsonl)
# Если вы запускаете это в той же папке, где лежит manifest.jsonl, можно оставить "."
WORK_DIR = Path(".").resolve()
MANIFEST_PATH = WORK_DIR / ".." / "seals_images_jpeg_downloaded" / "manifest.jsonl" # Укажите актуальный путь
# Если работаем с оригинальным синтетическим манифестом, раскомментируйте:
# MANIFEST_PATH = WORK_DIR / "runs" / "synthetic_v2" / "manifest.jsonl"

HOST = "127.0.0.1"
PORT = 5005

app = Flask(__name__)

# ===========================
# HTML TEMPLATE (Single Page App)
# ===========================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paleo-Hebrew Labeler</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; height: 100vh; display: flex; flex-direction: column; background: #1e1e1e; color: #ddd; }
        header { background: #2d2d2d; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #444; }
        h1 { margin: 0; font-size: 1.2rem; color: #61dafb; }
        
        #main-container { display: flex; flex: 1; overflow: hidden; }
        
        /* Canvas Area */
        #canvas-wrapper { flex: 1; background: #121212; display: flex; justify-content: center; align-items: center; overflow: auto; position: relative; }
        canvas { box-shadow: 0 0 20px rgba(0,0,0,0.5); cursor: crosshair; }
        
        /* Sidebar */
        #sidebar { width: 350px; background: #252526; padding: 20px; display: flex; flex-direction: column; border-left: 1px solid #444; overflow-y: auto; }
        
        .panel { background: #333; padding: 15px; border-radius: 6px; margin-bottom: 15px; }
        .label { font-size: 0.8rem; color: #aaa; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
        
        .hebrew-text { font-family: 'Times New Roman', serif; font-size: 2rem; direction: rtl; text-align: center; color: #fff; background: #000; padding: 10px; border-radius: 4px; letter-spacing: 2px; }
        
        .control-group { margin-top: 10px; display: flex; gap: 10px; }
        button { flex: 1; padding: 10px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; transition: 0.2s; }
        button:hover { opacity: 0.9; }
        .btn-primary { background: #0e639c; color: white; }
        .btn-success { background: #107c10; color: white; }
        .btn-danger { background: #c586c0; color: black; }
        .btn-nav { background: #444; color: white; }

        input[type="text"] { width: 100%; padding: 8px; font-size: 1.2rem; text-align: center; background: #1e1e1e; border: 1px solid #555; color: white; margin-top: 5px; }
        
        #box-list { list-style: none; padding: 0; margin: 0; max-height: 200px; overflow-y: auto; }
        #box-list li { padding: 8px; border-bottom: 1px solid #444; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }
        #box-list li:hover { background: #3e3e42; }
        #box-list li.active { background: #094771; }
        
        .badge { background: #555; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; }
        
        #status-bar { padding: 5px 20px; background: #007acc; color: white; font-size: 0.9rem; display: flex; justify-content: space-between; }
    </style>
</head>
<body>

<header>
    <h1>Paleo-Hebrew YOLO Annotator</h1>
    <div id="file-info">Loading...</div>
</header>

<div id="main-container">
    <div id="canvas-wrapper">
        <canvas id="editor"></canvas>
    </div>
    
    <div id="sidebar">
        <div class="panel">
            <div class="label">Manifest Text (Reference)</div>
            <div id="ref-text" class="hebrew-text">...</div>
        </div>

        <div class="panel">
            <div class="label">Selected Box Label</div>
            <input type="text" id="char-input" placeholder="Type char..." maxlength="10">
            <div style="font-size: 0.8rem; color: #888; margin-top: 5px;">Tip: Click box, type letter, press Enter.</div>
            <div class="control-group">
                 <button class="btn-danger" onclick="deleteSelectedBox()">Delete Box (Del)</button>
            </div>
        </div>

        <div class="panel" style="flex:1; display:flex; flex-direction:column;">
            <div class="label">Boxes (<span id="box-count">0</span>)</div>
            <ul id="box-list"></ul>
        </div>

        <div class="control-group">
            <button class="btn-nav" onclick="nav(-1)">Previous (Left)</button>
            <button class="btn-success" onclick="saveAndNav(1)">Save & Next (Right)</button>
        </div>
        <div class="control-group">
            <button class="btn-primary" onclick="saveData()">Save Only (Ctrl+S)</button>
        </div>
    </div>
</div>

<div id="status-bar">
    <span id="status-msg">Ready</span>
    <span id="progress-indicator">0 / 0</span>
</div>

<script>
    let canvas = document.getElementById('editor');
    let ctx = canvas.getContext('2d');
    let currentData = null;
    let currentIndex = 0;
    let totalItems = 0;
    
    // State
    let imageObj = new Image();
    let boxes = []; // {x, y, w, h, label, id}
    let selectedBoxIndex = -1;
    let isDragging = false;
    let dragStart = {x:0, y:0};
    let scale = 1.0;
    let offset = {x:0, y:0}; // For panning (optional, currently centered)

    // Interaction State
    let mode = 'idle'; // idle, drawing, moving, resizing
    let activeHandle = null; 

    // API
    async function loadData(index) {
        try {
            const response = await fetch(`/api/get_item/${index}`);
            const data = await response.json();
            
            if (data.error) {
                alert("End of list or error.");
                return;
            }

            currentData = data;
            currentIndex = data.index;
            totalItems = data.total;
            
            document.getElementById('file-info').textContent = `${data.uid} (${data.width}x${data.height})`;
            document.getElementById('progress-indicator').textContent = `${currentIndex + 1} / ${totalItems}`;
            
            // Set Ref text
            let refT = "";
            if (data.gt && data.gt.hebrew) refT = data.gt.hebrew;
            else if (data.gt && data.text) refT = data.text;
            else if (data.meta && data.meta.text) refT = data.meta.text;
            document.getElementById('ref-text').textContent = refT;

            // Load Boxes
            boxes = [];
            if (data.gt && data.gt.bboxes) {
                const chars = data.gt.chars || [];
                data.gt.bboxes.forEach((b, i) => {
                    // b is [x1, y1, x2, y2]
                    boxes.push({
                        x: b[0], y: b[1], w: b[2]-b[0], h: b[3]-b[1],
                        label: chars[i] || "?",
                        id: Date.now() + i
                    });
                });
            }

            // Load Image
            imageObj.onload = () => {
                fitCanvas();
                draw();
            };
            imageObj.src = `/api/image?path=${encodeURIComponent(data.image_path)}`;
            
            renderBoxList();
            setStatus("Loaded.");

        } catch (e) {
            console.error(e);
            setStatus("Error loading data");
        }
    }

    async function saveData() {
        if (!currentData) return;
        
        setStatus("Saving...");
        
        // Convert boxes back to [x1, y1, x2, y2] and char list
        const bboxes = boxes.map(b => [Math.round(b.x), Math.round(b.y), Math.round(b.x+b.w), Math.round(b.y+b.h)]);
        const chars = boxes.map(b => b.label);
        
        const payload = {
            index: currentIndex,
            uid: currentData.uid,
            bboxes: bboxes,
            chars: chars
        };

        try {
            await fetch('/api/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            setStatus("Saved.");
            return true;
        } catch(e) {
            alert("Failed to save!");
            return false;
        }
    }

    async function nav(dir) {
        const nextIdx = currentIndex + dir;
        if (nextIdx < 0 || nextIdx >= totalItems) return;
        loadData(nextIdx);
    }

    async function saveAndNav(dir) {
        const ok = await saveData();
        if (ok) nav(dir);
    }

    // --- Canvas Logic ---
    function fitCanvas() {
        const wrap = document.getElementById('canvas-wrapper');
        const availW = wrap.clientWidth - 40;
        const availH = wrap.clientHeight - 40;
        
        const imgRatio = imageObj.width / imageObj.height;
        const screenRatio = availW / availH;

        if (imgRatio > screenRatio) {
            // Limited by width
            scale = availW / imageObj.width;
        } else {
            // Limited by height
            scale = availH / imageObj.height;
        }
        
        canvas.width = imageObj.width * scale;
        canvas.height = imageObj.height * scale;
    }

    function getMousePos(evt) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (evt.clientX - rect.left) / scale,
            y: (evt.clientY - rect.top) / scale
        };
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.scale(scale, scale);
        
        // Draw Image
        ctx.drawImage(imageObj, 0, 0);

        // Draw Boxes
        boxes.forEach((b, idx) => {
            const isSel = (idx === selectedBoxIndex);
            
            ctx.beginPath();
            ctx.lineWidth = isSel ? 3 : 2;
            ctx.strokeStyle = isSel ? '#00ff00' : '#ff0000'; // Green if selected, red otherwise
            ctx.fillStyle = isSel ? 'rgba(0, 255, 0, 0.1)' : 'rgba(255, 0, 0, 0)';
            
            ctx.rect(b.x, b.y, b.w, b.h);
            ctx.stroke();
            ctx.fill();

            // Label
            if (b.label) {
                ctx.fillStyle = isSel ? '#00ff00' : '#ff0000';
                ctx.font = "20px Arial";
                ctx.fillText(b.label, b.x, b.y - 5);
            }
        });

        // Drawing new box
        if (mode === 'drawing') {
            const w = dragStart.currX - dragStart.x;
            const h = dragStart.currY - dragStart.y;
            ctx.strokeStyle = 'yellow';
            ctx.lineWidth = 2;
            ctx.strokeRect(dragStart.x, dragStart.y, w, h);
        }

        ctx.restore();
    }

    // --- Events ---
    canvas.addEventListener('mousedown', e => {
        const m = getMousePos(e);
        
        // Check selection
        // Simple hit test (last drawn on top)
        let hit = -1;
        for (let i = boxes.length - 1; i >= 0; i--) {
            const b = boxes[i];
            if (m.x >= b.x && m.x <= b.x + b.w && m.y >= b.y && m.y <= b.y + b.h) {
                hit = i;
                break;
            }
        }

        if (hit !== -1) {
            selectBox(hit);
            mode = 'moving'; // Simplified: drag to move whole box
            dragStart = {x: m.x, y: m.y, origX: boxes[hit].x, origY: boxes[hit].y};
        } else {
            // Start drawing
            selectBox(-1);
            mode = 'drawing';
            dragStart = {x: m.x, y: m.y, currX: m.x, currY: m.y};
        }
    });

    canvas.addEventListener('mousemove', e => {
        const m = getMousePos(e);
        
        if (mode === 'drawing') {
            dragStart.currX = m.x;
            dragStart.currY = m.y;
            draw();
        } else if (mode === 'moving' && selectedBoxIndex !== -1) {
            const dx = m.x - dragStart.x;
            const dy = m.y - dragStart.y;
            boxes[selectedBoxIndex].x = dragStart.origX + dx;
            boxes[selectedBoxIndex].y = dragStart.origY + dy;
            draw();
        }
    });

    canvas.addEventListener('mouseup', e => {
        if (mode === 'drawing') {
            const x = Math.min(dragStart.x, dragStart.currX);
            const y = Math.min(dragStart.y, dragStart.currY);
            const w = Math.abs(dragStart.currX - dragStart.x);
            const h = Math.abs(dragStart.currY - dragStart.y);

            if (w > 5 && h > 5) {
                boxes.push({x, y, w, h, label: '', id: Date.now()});
                selectBox(boxes.length - 1);
            }
        }
        mode = 'idle';
        draw();
        renderBoxList();
    });

    // --- UI Logic ---
    function selectBox(idx) {
        selectedBoxIndex = idx;
        const inp = document.getElementById('char-input');
        if (idx !== -1) {
            inp.value = boxes[idx].label || '';
            inp.focus();
        } else {
            inp.value = '';
            inp.blur();
        }
        draw();
        renderBoxList();
    }

    function renderBoxList() {
        const ul = document.getElementById('box-list');
        ul.innerHTML = '';
        document.getElementById('box-count').innerText = boxes.length;
        
        boxes.forEach((b, i) => {
            const li = document.createElement('li');
            li.innerHTML = `<span>Box ${i+1}</span> <span class="badge">${b.label || '?'}</span>`;
            if (i === selectedBoxIndex) li.className = 'active';
            li.onclick = () => selectBox(i);
            ul.appendChild(li);
        });
    }

    function deleteSelectedBox() {
        if (selectedBoxIndex === -1) return;
        boxes.splice(selectedBoxIndex, 1);
        selectBox(-1);
    }

    // Input binding
    document.getElementById('char-input').addEventListener('input', e => {
        if (selectedBoxIndex !== -1) {
            boxes[selectedBoxIndex].label = e.target.value;
            draw(); // redraw to update label on canvas
            renderBoxList();
        }
    });

    // Hotkeys
    document.addEventListener('keydown', e => {
        if (e.target.tagName === 'INPUT') {
            if (e.key === 'Enter') {
                e.target.blur(); // Unfocus
            }
            return; 
        }

        if (e.key === 'Delete') deleteSelectedBox();
        if (e.key === 'ArrowRight') saveAndNav(1);
        if (e.key === 'ArrowLeft') nav(-1);
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            saveData();
        }
    });

    function setStatus(msg) {
        document.getElementById('status-msg').textContent = msg;
    }

    window.addEventListener('resize', () => {
        if(imageObj.src) fitCanvas(); draw();
    });

    // Init
    loadData(0);

</script>
</body>
</html>
"""

# ===========================
# BACKEND LOGIC
# ===========================

manifest_data = []

def load_manifest():
    global manifest_data
    manifest_data = []
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        return
    
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                manifest_data.append(json.loads(line))
    print(f"Loaded {len(manifest_data)} records from {MANIFEST_PATH}")

def resolve_image_path(rec):
    # Logic from your previous scripts
    img = rec.get("image", {})
    
    # Try absolute path first
    ap = img.get("abs_path")
    if ap and os.path.exists(ap): return ap
    
    # Try relative path logic
    rp = img.get("rel_path")
    if rp:
        # Check relative to manifest dir
        cand1 = MANIFEST_PATH.parent / rp
        if cand1.exists(): return str(cand1)
        # Check relative to work dir
        cand2 = WORK_DIR / rp
        if cand2.exists(): return str(cand2)
    
    # Fallback path
    p = img.get("path")
    if p and os.path.exists(p): return p
    
    return None

# API ROUTES
@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/api/get_item/<int:index>')
def get_item(index):
    if index < 0 or index >= len(manifest_data):
        return jsonify({"error": "Index out of bounds"}), 404
    
    rec = manifest_data[index]
    img_path = resolve_image_path(rec)
    
    if not img_path:
        # If image missing, return metadata but handle error in frontend or skip
        return jsonify({"error": "Image file not found", "uid": rec.get("uid")})

    # Prepare safe data for frontend
    # Use existing GT if available, otherwise fallback
    gt = rec.get("gt", {})
    text = gt.get("hebrew") or rec.get("text", "")
    
    # Handle image dimensions (needed for scaling)
    # We will let frontend handle actual dimensions via image load, 
    # but send meta width/height if available
    w = rec.get("image", {}).get("width", 0)
    h = rec.get("image", {}).get("height", 0)

    return jsonify({
        "index": index,
        "total": len(manifest_data),
        "uid": rec.get("uid"),
        "image_path": img_path,
        "gt": gt,
        "text": text,
        "meta": rec.get("meta", {}),
        "width": w,
        "height": h
    })

@app.route('/api/image')
def serve_image():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return "Image not found", 404
    return send_file(path)

@app.route('/api/save', methods=['POST'])
def save_annotation():
    data = request.json
    idx = data.get('index')
    bboxes = data.get('bboxes')
    chars = data.get('chars')
    
    if idx is None or idx < 0 or idx >= len(manifest_data):
        return jsonify({"error": "Invalid index"}), 400

    # Update in memory
    rec = manifest_data[idx]
    
    # Ensure gt structure exists
    if "gt" not in rec or rec["gt"] is None:
        rec["gt"] = {}
        
    rec["gt"]["bboxes"] = bboxes
    rec["gt"]["chars"] = chars
    
    # Reconstruct the 'hebrew' text from chars if needed, 
    # OR assume user only fixes boxes and chars correspond to text.
    # For now, we just save the bboxes/chars.
    
    # Update on Disk (Atomic-ish write)
    try:
        # Rewrite the whole file (safest for jsonl integrity, though slow for massive files)
        # For 220k lines, rewriting every save is bad.
        # Better: We keep memory sync, and have a 'Backup' button or save periodically?
        # For this tool, let's just append to a NEW file and periodic merge, 
        # OR just rewrite. 220k lines is about 50MB. Rewriting takes < 0.5s. It's fine.
        
        with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
            for item in manifest_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        return jsonify({"status": "ok"})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

def run_app():
    load_manifest()
    print(f"🚀 Starting Labeling Tool at http://{HOST}:{PORT}")
    print(f"📂 Manifest: {MANIFEST_PATH}")
    # Open browser automatically
    threading.Timer(1.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)

if __name__ == '__main__':
    run_app()