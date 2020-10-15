import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
# from utils.utils import *
from utils.general import (
    check_img_size, non_max_suppression,non_max_suppression_test, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized,initialize_weights
from modelsori import *

def detect(number_person):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    half=False

    # Load model
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    # imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # model=torch.load(weights, map_location=device)['model'].float().eval()

    model = Darknet('cfg/prune_0.8_yolov3-spp.cfg', (opt.img_size, opt.img_size)).to(device)
    initialize_weights(model)
    model.load_state_dict(torch.load('weights/prune_0.8_yolov3-spp-ultralytics.pt')['model'])
    model.eval()
    stride = [8, 16, 32]
    imgsz = check_img_size(imgsz, s=max(stride))  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    names = ['1', '2']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for dirs in os.listdir(source):
        # if dirs !='WH':
        #     continue
        src = os.path.join(source, dirs)
        save_img = True
        dataset = LoadImages(src, img_size=imgsz)
        for path, img, im0s, vid_cap in dataset:
            # if os.path.basename(path)!='2_31746253093C100D_2018-12-10-21-56-37-998_0_75_636_307_6.jpg':
            #     continue
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression_test(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
            #                                 agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                # save_path = str(Path(out) / Path(p).name)
                # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
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
                    results = [0, 0]
                    for *xyxy, conf, cls in det:
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if save_img or view_img:  # Add bbox to image
                            if names[int(cls)] == "1":
                                results[0] += 1
                            else:
                                results[1] += 1
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    if len(results) == 2:
                        if (results[0] > number_person) | (results[1] > number_person):
                            tmp = "err"
                        elif (results[0] == number_person) | (results[1] == number_person):
                            tmp = "corr"
                        else:
                            tmp = "miss"
                    elif len(results) == 1:
                        if (results[0] == number_person):
                            tmp = "corr"
                        elif (results[0] > number_person):
                            tmp = "err"
                        else:
                            tmp = "miss"
                    else:
                        tmp = "miss"

                    save_path = os.path.join(Path(out), dirs, tmp)#, tmp
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(os.path.join(save_path, Path(p).name), im0)
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

        if save_txt or save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + out)
            # if platform == 'darwin' and not opt.update:  # MacOS
            #     os.system('open ' + save_path)

        print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  #/home/lzm/Disk3T/1-FWS_data/TestData_image/TX2_Test_data/double_company
    parser.add_argument('--weights', nargs='+', type=str, default='/home/lishuang/Disk/remote/pycharm/yolov5/runs/last_s_prune.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/home/lishuang/Disk/shengshi_data/anti_tail_test_dataset/Data_of_each_scene', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='/home/lishuang/Disk/remote/pycharm/yolov5s_416_04_last', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
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
    number_person=1
    if os.path.basename(opt.source)=="Data_of_each_scene":
        number_person = 1
    elif os.path.basename(opt.source)=="double_company":
        number_person=2
    else:
        print("error image file!!!")
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect(number_person)
                strip_optimizer(opt.weights)
        else:
            detect(number_person)
