#train
python train_pytorch1.4_noprune.py --img 640 --batch 4 --epochs 100 --data ./data/coco128.yaml --cfg ./models/yolov5s.yaml --weights weights/yolov5s.pt --name s

python train_pytorch1.4_noprune.py --img 640 --batch 8 --epochs 100 --data ./data/hand.yaml --cfg ./models/yolov5s_hand.yaml --weights weights/yolov5s.pt --name s_hand

#train_for_prune
python train_pytorch1.4.py --img 640 --batch 16 --epochs 300 --data ./data/hand.yaml --cfg ./models/yolov5s_hand.yaml --weights weights/last_s_hand.pt --name s_to_prune -sr --s 0.001 --prune 1




#剪枝前的基础训练
#低版本pytorch
python train_pytorch1.4_noprune.py --img 640 --batch 8 --epochs 100 --data ./data/coco_hand.yaml --cfg ./models/yolov5s_hand.yaml --weights weights/yolov5s_v2.pt --name s_hand
#高版本pytorch，支持自带的AMP
python train.py --img 640 --batch 8 --epochs 100 --data ./data/coco_hand.yaml --cfg ./models/yolov5s_hand.yaml --weights weights/yolov5s_v2.pt --name s_hand

#稀疏训练
python train_pytorch1.4_sparsity.py --img 640 --batch 8 --epochs 300 --data ./data/coco_hand.yaml --cfg ./models/yolov5s_hand.yaml --weights runs/exp1_s_hand/weights/last_s_hand.pt --name s_hand_sparsity -sr --s 0.001 --prune 1

#todo
#prune_finetune
python prune_finetune.py --img 640 --batch 8 --epochs 20 --data ./data/coco_hand.yaml --cfg ./cfg/prune_0.5_keep_0.01_8x_yolov5s_v2_hand.cfg --weights ./weights/prune_0.5_keep_0.01_8x_last_s_hand_sparsity.pt --name prunefinetune_hand_s

#prune model inference
python prune_detect.py --weights weights/last_prune_hand_s.pt --img  640 --conf 0.7 --save-txt --source inference/images