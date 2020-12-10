#train
python train_pytorch1.4_noprune.py --img 640 --batch 4 --epochs 100 --data ./data/coco128.yaml --cfg ./models/yolov5s.yaml --weights weights/yolov5s.pt --name s

python train_pytorch1.4_noprune.py --img 640 --batch 8 --epochs 100 --data ./data/hand.yaml --cfg ./models/yolov5s_hand.yaml --weights weights/yolov5s.pt --name s_hand

#train_for_prune
python train_pytorch1.4.py --img 640 --batch 16 --epochs 300 --data ./data/hand.yaml --cfg ./models/yolov5s_hand.yaml --weights weights/last_s_hand.pt --name s_to_prune -sr --s 0.001 --prune 1

#prune_finetune
python prune_finetune.py --img 640 --batch 8 --epochs 10 --data ./data/hand.yaml --cfg ./cfg/prune_0.8_keep_0.01_8x_yolov5s_hand.cfg --weights ./weights/prune_0.8_keep_0.01_8x_last_s_to_prune.pt --name prune_hand_s
