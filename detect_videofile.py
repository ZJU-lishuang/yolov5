import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
# from utils.utils import *
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized,initialize_weights
# from modelsori import *

def point_in_box(points,polygon):
    inside=False
    x,y=points
    xmin,ymin,xmax,ymax=polygon
    if x >xmin and x< xmax and y>ymin and y< ymax:
        inside=True
    return inside

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    half=False

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # model = Darknet('cfg/prune_0.8_yolov3-spp.cfg', (opt.img_size, opt.img_size)).to(device)
    # initialize_weights(model)
    # model.load_state_dict(torch.load('weights/prune_0.8_yolov3-spp-ultralytics.pt')['model'])
    # model.eval()
    # stride = [8, 16, 32]
    # imgsz = check_img_size(imgsz, s=max(stride))  # check img_size
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
    # if webcam:
    #     view_img = True
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz)
    # else:
    #     save_img = True
    #     dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # names = ['1', '2']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    videopath_list = (
        # '2020double_company',
        # '2020double_company_1',
        # 'child_79_company',
        # 'child_137_huaxia',
        # 'child_137_huaxia_1',
        # 'double_54_zhuhai',
        # 'double_54_zhuhai_1',
        # 'double_59_huaxiaxueyuan',
        # 'double_59_huaxiaxueyuan_1',
        # 'double_990_close_company',
        # 'double_beijing',
        # 'double_beijing_1',
        'single_28_huaxia',
        # 'single_28_huaxia_2',
        # 'single_897_yinchuan',
        # 'single_897_yinchuan_2',
        # 'single_1000_beijng_shoudu',
        # 'single_1000_guangzhjou',
        # 'single_1000_wuhan',
    )

    video_dir_pass=[
                    'single_1000_beijng_shoudu_kuan',       
                    'single_1000_wuhan_kuan', ]
#     video_path='/home/lishuang/Disk/shengshi_data/video_test_split_all/single_1000_beijng_shoudu_test_frame'
    video_dir_path='/home/lishuang/Disk/shengshi_data/video_test_split_all'
    video_paths=os.listdir(video_dir_path)
    for video_dir in video_paths:
        if video_dir  not in videopath_list:
            print(video_dir," pass")
            continue
        video_path=os.path.join(video_dir_path,video_dir)
#         if video_dir !='double_54_zhuhai':
#             continue

        csv_path = os.path.join(video_dir_path, 'video_test_csv', f'{video_dir}_video_cut.csv')
        # csv_path = os.path.join(os.path.join(videopath, ".."), f'{basedirname}_video_cut.csv')

        video_name = []
        video_name_dic = {}
        with open(csv_path) as f:
            lines = f.readlines()[1:]
            for line in lines:
                line = line.rstrip()
                items = line.split(',')
                video_name.append(items[1])
                video_name_dic[items[1]] = [items[2], items[3], items[4], items[5]]

        if os.path.isdir(video_path):
            video_files=os.listdir(video_path)
            alarmvideo_list = {}
            for video_file in video_files:
                if video_file!='616643FEF1380C0E_2019-10-19-11-37-49-812_passenger_00000061_2.mp4':
                    continue
                if video_file[:-4] not in video_name_dic:
                    continue
                videosource=os.path.join(video_path,video_file)
                # if len(os.listdir(videosource))==0:
                #     continue
                save_img = True
                view_img = True
                videodataset = LoadImages(videosource, img_size=imgsz)
                video_file, extension = os.path.splitext(video_file)
                alarmvideo_list[video_file] = 0
                frame_record = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                frame_num=0
                outvideo=str(Path(out) /video_dir/video_file)
                x1t, y1t, x2t, y2t = video_name_dic[video_file]
                ratio_width=1
                ratio_height=1
                x1t = int(x1t) * ratio_width
                x2t = int(x2t) * ratio_width
                y1t = int(y1t) * ratio_height
                y2t = int(y2t) * ratio_height

                if os.path.exists(outvideo):
                    shutil.rmtree(outvideo)  # delete output folder
                os.makedirs(outvideo)  # make new output folder
                for path, img, im0s, vid_cap in videodataset:   #one video
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=opt.augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                               agnostic=opt.agnostic_nms)
                    t2 = time_synchronized()

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

                        save_path = str(Path(out) /video_dir/ Path(p).name)
                        # txt_path = str(Path(out) /video_dir/video_file/ Path(p).stem) + ('_%g' % videodataset.frame if videodataset.mode == 'video' else '')
                        txt_path = str(Path(out) / video_dir / video_file /str(videodataset.frame))
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
                                # if cls == 0:
                                #     # label = 'person'
                                #     boxnum += 1
                                #     boxnumbody += 1
                                # elif cls == 1:
                                #     # label = 'head'
                                #     boxnumhead += 1
                                # if point_in_box(box_center, [x1, y1, x2, y2]):
                                #     boxnumhead += 1 * person_result['class'] == 2
                                #     boxnumbody += 1 * person_result['class'] == 1
                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                    with open(txt_path + '.txt', 'a') as f:
                                        x0, y0, w0, h0 = xywh
                                        h, w = im0.shape[:2]
                                        x0 *= w
                                        y0 *= h
                                        w0 *= w
                                        h0 *= h
                                        x1 = x0 - w0 / 2
                                        y1 = y0 - h0 / 2
                                        if point_in_box([x0,y0], [x1t, y1t, x2t, y2t]):
                                            boxnumhead += 1 * cls == 1
                                            boxnumbody += 1 * cls == 0
                                        f.write(('%s ' + '%.2g ' + '%d ' * 3 + '%d' + '\n') % (
                                        names[int(cls)], conf, x1, y1, w0, h0))  # label format
                                #                             f.write(('%ss '+'%.2g ' * 5 + '\n') % (names[int(cls)], conf,*xywh))  # label format

                                if save_img or view_img:  # Add bbox to image
                                    label = '%s %.2f' % (names[int(cls)], conf)
                                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                        # Print time (inference + NMS)
                        print('%sDone. (%.3fs)' % (s, t2 - t1))

                        # Stream results
                        if view_img:
                            cv2.imshow(p, im0)
                            if cv2.waitKey(1) == ord('q'):  # q to quit
                                raise StopIteration

                        # Save results (image with detections)
                        if save_img:
                            if videodataset.mode == 'images':
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
                                image_path=os.path.join(outvideo,str(videodataset.frame)+'.jpg')
                                cv2.imwrite(image_path, im0)
                        if boxnumbody > 1 or boxnumhead > 1:
                            frame_record[frame_num % 10] = 1
                        else:
                            frame_record[frame_num % 10] = 0
                        frame_num+=1
                        if alarmvideo_list[video_file] ==0 and sum(frame_record) >7:
                            alarmvideo_list[video_file] =1
                            image_path = os.path.join(outvideo, str(videodataset.frame) + '_alarmvideo.jpg')
                            cv2.imwrite(image_path, im0)
            file_data=""
            for single_video in alarmvideo_list:
                file_data += str(single_video) + ', value: ' + str(alarmvideo_list[single_video]) + '\n'
            with open(f'{os.path.basename(video_path)}_video_result_{opt.conf_thres}.txt', 'a') as f:
                f.write(file_data)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/lishuang/Disk/remote/pycharm/yolov5/runs/last_s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output_video', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect()
                create_pretrained(opt.weights, opt.weights)
        else:
            detect()
