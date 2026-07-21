#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""end2end_infer (wrapper)

Compatibility wrapper: older scripts expect `end2end_infer.py`.
Actual implementation lives in `end_2_end_infer.py`.
"""

from end_2_end_infer import main

if __name__ == "__main__":
    main()
