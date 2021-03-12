#!/bin/sh
python detect_fangweisui.py --weights runs/exp4_l/weights/best_l.pt --img 640 --conf 0.8 --save-txt
