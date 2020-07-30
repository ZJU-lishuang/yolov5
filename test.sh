#!/bin/sh
python detect_fangweisui.py --weights runs/exp1_m/weights/best.pt --img 512 --conf 0.8 --save-txt
