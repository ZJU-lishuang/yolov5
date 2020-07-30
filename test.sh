#!/bin/sh
python detect.py --source ../yolov5data/fangweisui/images/tx2_train_data_with_xml/T_2018_06_06_1_company_2 --output ./inference/T_2018_06_06_1_company_2 --weights runs/exp1_m/weights/last.pt --img 512 --conf 0.8 --save-txt

python detect.py --source ../yolov5data/fangweisui/images/tx2_train_data_with_xml/T_2018_07_09_hxxy_2 --output ./inference/T_2018_07_09_hxxy_2 --weights runs/exp1_m/weights/last.pt --img 512 --conf 0.8 --save-txt

python detect.py --source ../yolov5data/fangweisui/images/tx2_train_data_with_xml/T_2018_07_09_hxxy_out_2 --output ./inference/T_2018_07_09_hxxy_out_2 --weights runs/exp1_m/weights/last.pt --img 512 --conf 0.8 --save-txt

python detect.py --source ../yolov5data/fangweisui/images/tx2_train_data_with_xml/T_2018_07_12_1_company_1 --output ./inference/T_2018_07_12_1_company_1 --weights runs/exp1_m/weights/last.pt --img 512 --conf 0.8 --save-txt

python detect.py --source ../yolov5data/fangweisui/images/tx2_train_data_with_xml/T_2018_07_17_hxxy_1 --output ./inference/T_2018_07_17_hxxy_1 --weights runs/exp1_m/weights/last.pt --img 512 --conf 0.8 --save-txt
