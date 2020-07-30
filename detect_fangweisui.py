import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = True
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz)
    # else:
    #     save_img = True
    #     dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once


    img_dir_path='/home/lishuang/Disk/remote/pycharm/yolov5data/fangweisui/images/tx2_train_data_with_xml'
    answerdict = {"T_2018_06_06_1_company_2": 2,
                "T_2018_07_09_hxxy_2": 2,
                "T_2018_07_09_hxxy_out_2": 2,
                "T_2018_07_12_1_company_1": 1,
                "T_2018_07_17_hxxy_1": 1,
                "T_2018_07_19_child_1": 1,
                "T_2018_07_19_child_2": 2,
                "T_2018_07_19_child_3": 3,
                "T_2018_07_19_child_out_1": 1,
                "T_2018_07_19_child_out_2": 2,
                "T_child_2017-04-22_3_35_2": 2,
                "T_child_company_2018_11_29_2": 2,
                "T_gangzhuao_xianchang_1": 1,
                "T_shanghai_hongqiao_xianchang_test": 1,
                }
    file_data = ""
    file_data += "filename,0,1,2,3,miss,correct,error\n"

    num_zero = {}
    num_one = {}
    num_two = {}
    num_three = {}
    num_head = {}
    num_body = {}
    img_paths=os.listdir(img_dir_path)
    for img_dir in img_paths:
        img_path=os.path.join(img_dir_path,img_dir)
        if os.path.isdir(img_path):

            save_img = True
            imgdataset = LoadImages(img_path, img_size=imgsz)

            outimg=str(Path(out) /img_dir)
            if os.path.exists(outimg):
                shutil.rmtree(outimg)  # delete output folder
            os.makedirs(outimg)  # make new output folder

            outimg=str(Path(out) /img_dir/"0")
            if os.path.exists(outimg):
                shutil.rmtree(outimg)  # delete output folder
            os.makedirs(outimg)  # make new output folder

            outimg=str(Path(out) /img_dir/"1")
            if os.path.exists(outimg):
                shutil.rmtree(outimg)  # delete output folder
            os.makedirs(outimg)  # make new output folder

            outimg=str(Path(out) /img_dir/"2")
            if os.path.exists(outimg):
                shutil.rmtree(outimg)  # delete output folder
            os.makedirs(outimg)  # make new output folder

            outimg=str(Path(out) /img_dir/"3")
            if os.path.exists(outimg):
                shutil.rmtree(outimg)  # delete output folder
            os.makedirs(outimg)  # make new output folder

            
            for path, img, im0s, vid_cap in imgdataset:   #one video
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = torch_utils.time_synchronized()
                    pred = model(img, augment=opt.augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                               agnostic=opt.agnostic_nms)
                    t2 = torch_utils.time_synchronized()

                    # Apply Classifier
                    if classify:
                        pred = apply_classifier(pred, modelc, img, im0s)

                    boxnum = 0
                    boxnumbody = 0
                    boxnumhead = 0
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        if webcam:  # batch_size >= 1
                            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                        else:
                            p, s, im0 = path, '', im0s

                        save_path = str(Path(out) /img_dir/ Path(p).name)
                        s += '%gx%g ' % img.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                        if det is not None and len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += '%g %ss, ' % (n, names[int(c)])  # add to string


                            # Write results
                            for *xyxy, conf, cls in det:
                                if cls == 0:
                                    # label = 'person'
                                    boxnum += 1
                                    boxnumbody += 1
                                elif cls == 1:
                                    # label = 'head'
                                    boxnumhead += 1
                                

                                if save_img or view_img:  # Add bbox to image
                                    label = '%s %.2f' % (names[int(cls)], conf)
                                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                        if img_dir not in num_zero:
                            num_zero[img_dir] = 0
                        if img_dir not in num_one:
                            num_one[img_dir] = 0
                        if img_dir not in num_two:
                            num_two[img_dir] = 0
                        if img_dir not in num_three:
                            num_three[img_dir] = 0
                        boxnum=max(boxnumbody,boxnumhead)
                        if boxnum == 0:
                            num_zero[img_dir] = num_zero[img_dir] + 1
                            save_path = str(Path(out) /img_dir/ "0"/Path(p).name)
                        elif boxnum == 1:
                            num_one[img_dir] = num_one[img_dir] + 1
                            save_path = str(Path(out) /img_dir/ "1"/Path(p).name)
                        elif boxnum == 2:
                            num_two[img_dir] = num_two[img_dir] + 1
                            save_path = str(Path(out) /img_dir/ "2"/Path(p).name)
                        else:
                            num_three[img_dir] = num_three[img_dir] + 1
                            save_path = str(Path(out) /img_dir/ "3"/Path(p).name)

                        # Print time (inference + NMS)
                        print('%sDone. (%.3fs)' % (s, t2 - t1))

                        # Stream results
                        if view_img:
                            cv2.imshow(p, im0)
                            if cv2.waitKey(1) == ord('q'):  # q to quit
                                raise StopIteration

                        # Save results (image with detections)
                        if save_img:
                            if imgdataset.mode == 'images':
                                cv2.imwrite(save_path, im0)
                            else:
                                if vid_path != save_path:  # new video
                                    vid_path = save_path
                                    if isinstance(vid_writer, cv2.VideoWriter):
                                        vid_writer.release()  # release previous video writer

                                    fourcc = 'mp4v'  # output video codec
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                                vid_writer.write(im0)

    for key in img_paths:
        num_error = 0
        num_correct = 0
        num_miss = 0
    if answerdict[key] == 1:
        num_error = num_two[key] + num_three[key]
        num_correct = num_one[key]
        num_miss = num_zero[key]
    elif answerdict[key] == 2:
        num_error = num_three[key]
        num_correct = num_two[key]
        num_miss = num_zero[key] + num_one[key]
    elif answerdict[key] == 3:
        num_error = 0
        num_correct = num_three[key]
        num_miss = num_zero[key] + num_one[key] + num_two[key]
    elif answerdict[key] == 0:
        num_error = num_one[key] + num_two[key] + num_three[key]
        num_correct = num_zero[key]
        num_miss = 0
    file_data += key + ",{},{},{},{},{},{},{},{},{}\n".format(num_zero[key], num_one[key], num_two[key], num_three[key],
                                                        num_miss,
                                                        num_correct, num_error)

    with open('yolov5-fangweisui.csv', 'w') as f:
        f.write(file_data)
    
    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
