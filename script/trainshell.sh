#!/bin/sh
python train.py --img 512 --batch 8 --epochs 100 --data ./data/coco_fangweisui.yaml --cfg ./models/yolov5m_fangweisui.yaml --weights 'weights/yolov5m.pt' --name m


