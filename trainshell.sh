#!/bin/sh
python train.py --img 512 --batch 8 --epochs 100 --data ./data/coco_fangweisui.yaml --cfg ./models/yolov5m_fangweisui.yaml --weights 'weights/yolov5m.pt' --name m




#添加数据增加，调整过滤规则，消除减半的矩形框
python train.py --img 512 --batch 4 --epochs 300 --data ./data/coco_fangweisui.yaml --cfg ./models/yolov5m_fangweisui.yaml --weights 'weights/yolov5m.pt' --name m

python train.py --img 640 --batch 4 --epochs 300 --data ./data/coco_fangweisui.yaml --cfg ./models/yolov5l_fangweisui.yaml --weights 'weights/yolov5l.pt' --name l

python train.py --img 640 --batch 4 --epochs 50 --data ./data/coco_fangweisui.yaml --cfg ./models/yolov5l_fangweisui.yaml --weights 'weights/yolov5l.pt' --name l


python train.py --img 640 --batch 2 --epochs 300 --data ./data/coco_fangweisui.yaml --cfg ./models/yolov5x_fangweisui.yaml --weights 'weights/yolov5x.pt' --name x




#去除视频标注的数据集
python train.py --img 640 --batch 4 --epochs 50 --data ./data/coco_fangweisui.yaml --cfg ./models/yolov5l_fangweisui.yaml --weights 'weights/yolov5l.pt' --name l


#TODO 恢复数据增强过滤矩形框的阈值
