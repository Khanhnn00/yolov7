import requests
import numpy as np
import time
from datetime import datetime
import os
import urllib3
import torch
from pathlib import Path
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import cv2
import random
import sys
import shutil
from datetime import datetime
sys.path.insert(1, '/opt/work/yolov7')  

from models.experimental import attempt_load
from utils.datasets import LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized

def check_valid_video_folder(url):
    if not os.path.exists(url):
        os.makedirs(url)

def download_video(video_url):
    # header = random_headers()
    # response = requests.get(video_url, headers=header, stream=True, verify=False, timeout=5)
    if not os.path.exists('video_temp'):
        os.mkdir('video_temp')
    try:
        response = requests.get(video_url, verify=False)
        # prefix = "video_temp/" + datetime.today().strftime("%Y_%m_%d_%H_%M_%S_")
        prefix = "video_temp/"
        vid_name = prefix + video_url.split('/')[-1]
        print("vid_name: ", vid_name)
        with open(vid_name, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        print("download_video", e)
        vid_name = None
    return vid_name

def predict_batch(model, imgs):
        bboxes = model(imgs)[0]
        return bboxes

def preprocess_img(img, stride):
    h0, w0 = img.shape[0], img.shape[1]
    r = 640 / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    
    img = letterbox(img, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img/255.

def predict_video_path(path, model, stride, device, cf, save_dir):
    # video_name = path
    # vid_name = path.split('/')[-1].split('.')[0]
    video_name = download_video(path)
    vid_name = video_name.split('/')[-1].split('.')[0]
    # print(vid_name)
    check_valid_video_folder(save_dir)
    st = time.time()
    vidcap = cv2.VideoCapture(video_name)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    frame_count = round(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = round(frame_count/fps)
    # print(duration)
    # print(fps)
    if not fps:
        fps = 25
    # success,image = vidcap.read()
    count = 0
    frames = []
    im0s = []
    preprocess_time = 0
    while vidcap.isOpened():
        success,image = vidcap.read()
        if isinstance(image, np.ndarray):
            if count % fps == 0:
                sub_st = time.time()
                frames.append(preprocess_img(image, stride))
                sub_end = time.time()
                preprocess_time += sub_end - sub_st
                im0s.append(image)
                # cv2.imwrite("{}/frame{}.jpg".format(this_dir, count), image)     # save frame as JPEG file      
            count += 1
        else:
            break
    vidcap.release()
    end = time.time()
    time_extract = round(end - st)

    # print("len of dataset: {}".format(len(frames)))
    half = device != 'cpu'

    if half:
        model = model.half()

    label = "normal"

    ind = 0
    bs = 64
    st = time.time()
    hard_margin = 0
    soft_margin = 0
    cnt_total = 0
    cnt = 0
    while ind < len(frames):
        if ind + bs <= len(frames):
            batch = frames[ind:ind+bs]
        else:
            batch = frames[ind:ind+bs]
            batch += [frames[-1]]*bs
            batch = batch[:bs]

        # batch = batch/255.
        batch = np.array(batch)
        batch = torch.from_numpy(batch).to(device)
        batch = batch.half() if half else batch.float()  # uint8 to fp16/32

        # Inference
        t1 = time_synchronized()
        pred = predict_batch(model, batch)
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, cf.model['conf'], cf.model['iou'])
        t3 = time_synchronized()

        # Process detections
        
        thresh = 5
        
        for i, det in enumerate(pred): # 64 C H W
            # print(ind, i)
            # print(len(im0s)) # detections per image
            if ind + i >= len(im0s):
                pass
            else:
                im0 = im0s[ind+i]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                
                if len(det):
                    cnt_total += len(det)
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(frames[ind+i].shape[1:], det[:, :4], im0.shape).round()

                    # Print results

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if conf >= 0.55:
                            hard_margin += 1
                            cnt += 1
                        
                        if conf < 0.5:
                            soft_margin += 1


                        # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) # label format
                        with open('{}/result.txt'.format(save_dir), 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        f.close()

                        # Add bbox to image
                        label = f'{conf:.2f}'
                        # plot_one_box(xyxy, im0s[ind+i], label=label, color=colors[int(cls)], line_thickness=1)
                        plot_one_box(xyxy, im0, label=label, line_thickness=3)
                        cv2.imwrite('{}/{}.jpg'.format(save_dir, vid_name), im0)
                        # print("The image with the result is saved in: video_temp/{}.png".format(vid_name))
                    
        
        ind += bs

    end_time = time.time()
    # shutil.rmtree(this_dir)
    print('Extracting video costs: {}s'.format(time_extract))
    print('Detection costs: {}s to run over video {}.mp4 with length of {}s'.format(round(end_time - st), vid_name, duration))
    print('Total bboxes found: {}'.format(cnt_total))
    print('Total bboxes with c  onf > 0.55: {}'.format(hard_margin))
    print('Total bboxes with conf < 0.5: {}'.format(soft_margin))
    if cnt_total == 0:
        rating_hard = 0
        rating_soft = 0
    else:
        rating_hard = hard_margin/cnt_total
        rating_soft = soft_margin/cnt_total

    if cnt >= thresh:
        label = "ndl"
    else:
        label = "normal"

    print('Rating hard: {}'.format(rating_hard))
    print('Rating soft: {}'.format(rating_soft))
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open('{}/{}.txt'.format(save_dir, 'result'), 'a') as f:
        f.write('{} \n'.format(dt_string))
        f.write('Extracting video costs: {}s \n'.format(time_extract))
        f.write('Detection costs: {}s to run over video {}.mp4 with length of {}s \n'.format(round(end_time - st), vid_name, duration))
        f.write('Total bboxes found: {}, where {} bboxes have conf > 0.55 and {} bboxes have conf < 0.5 \n'.format(cnt_total, hard_margin, soft_margin))
        f.write('Rating: {} and {} \n'.format(rating_hard, rating_soft))
        f.write('Final label: {} \n'.format(label))
        f.write('\n')
        f.write('\n')
    f.close()

    
    return label, rating_hard, rating_soft, vid_name


def predict_video(path, model, stride, device, cf):
    video_name = download_video(path)
    vid_name = path.split('/')[-1].split('.')[0]
    print(vid_name)
    save_dir = 'video_temp'
    check_valid_video_folder(save_dir)
    st = time.time()
    vidcap = cv2.VideoCapture(video_name)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = round(frame_count/fps)
    print(duration)
    print(fps)
    if not fps:
        fps = 25
    # success,image = vidcap.read()
    count = 0
    frames = []
    im0s = []
    preprocess_time = 0
    while vidcap.isOpened():
        success,image = vidcap.read()
        if isinstance(image, np.ndarray):
            if count % fps == 0:
                sub_st = time.time()
                frames.append(preprocess_img(image, stride))
                sub_end = time.time()
                preprocess_time += sub_end - sub_st
                im0s.append(image)
                # cv2.imwrite("{}/frame{}.jpg".format(this_dir, count), image)     # save frame as JPEG file      
            count += 1
        else:
            break
    vidcap.release()
    os.remove(video_name)
    end = time.time()
    time_extract = round(end - st)

    # print("len of dataset: {}".format(len(frames)))
    half = device != 'cpu'

    if half:
        model = model.half()

    label = "normal"

    ind = 0
    bs = 64
    st = time.time()
    while ind < len(frames):
        if ind + bs <= len(frames):
            batch = frames[ind:ind+bs]
        else:
            batch = frames[ind:ind+bs]
            batch += [frames[-1]]*bs
            batch = batch[:bs]

        # batch = batch/255.
        batch = np.array(batch)
        batch = torch.from_numpy(batch).to(device)
        batch = batch.half() if half else batch.float()  # uint8 to fp16/32

        # Inference
        t1 = time_synchronized()
        pred = predict_batch(model, batch)
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, cf.model['conf'], cf.model['iou'])
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred): # 64 C H W
            # print(ind, i)
            # print(len(im0s)) # detections per image
            if ind + i >= len(im0s):
                pass
            else:
                im0 = im0s[ind+i]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(frames[ind+i].shape[1:], det[:, :4], im0.shape).round()

                    # Print results

                    # Write results
                    for *xyxy, conf, cls in reversed(det):

                        # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) # label format
                        with open('video_temp/{}.txt'.format(vid_name), 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        f.close()

                    # Add bbox to image
                        label = f'{conf:.2f}'
                        # plot_one_box(xyxy, im0s[ind+i], label=label, color=colors[int(cls)], line_thickness=1)
                        plot_one_box(xyxy, im0, line_thickness=3)
                    cv2.imwrite('video_temp/{}.png'.format(vid_name), im0)
                    print("The image with the result is saved in: video_temp/{}.png".format(vid_name))
        
        ind += bs
    
    end_time = time.time()
    # shutil.rmtree(this_dir)
    print('Extracting video costs: {}s'.format(time_extract))
    print('Preprocessing image costs: {}s'.format(round(preprocess_time)))
    print('Detection costs: {}s to run over video {}.mp4 with length of {}s'.format(round(end_time - st), vid_name, duration))
    with open('video_temp/{}.txt'.format(vid_name), 'a') as f:
        f.write('Extracting video costs: {}s \n'.format(time_extract))
        f.write('Detection costs: {}s to run over video {}.mp4 with length of {}s'.format(round(end_time - st), vid_name, duration))
    f.close()
    return label