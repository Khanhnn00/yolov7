import io
import torch
import os
import sys
import uvicorn
import numpy as np
import nest_asyncio
import inspect
import cv2
from pathlib import Path
import random

from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from cvlib.object_detection import draw_bbox

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

from models.experimental import attempt_load
from utils.datasets import LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized

app = FastAPI(title='Nine-dash-line')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
half = device != 'cpu'
save_img = True
save_txt = True

model = attempt_load('./best.pt', map_location=device)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
model.eval()
iou_thres=0.6
conf_thres = 0.5

if not os.path.exists('images_uploaded'):
    os.mkdir('images_uploaded')
    
if not os.path.exists('video_saved'):
    os.mkdir('video_saved')
    
save_dir = 'results_detect'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

@app.get("/")
def home():
    return "The Nine-dash-line API is running. Please head over to http://localhost:8000/docs."

# This endpoint handles all the logic necessary for the object detection to work.
# It requires the desired model and the image in which to perform object detection.
@app.post("/predict/image") 
def predict_img(file: UploadFile = File(...)):

    # 1. VALIDATE INPUT FILE
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    
    image_stream = io.BytesIO(file.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    h0, w0 = img.shape[0], img.shape[1]
    r = 640 / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    h1, w1 = img.shape[0], img.shape[1]
    # print(h1, w1)
    
    img = letterbox(img, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    
    ts_img = torch.from_numpy(img).to(device)
    ts_img = ts_img.unsqueeze(0)
    print(ts_img.shape)
    ts_img = ts_img.float()  # uint8 to fp16/32
    ts_img /= 255.0  # 0 - 255 to 0.0 - 1.0
    out, train_out = model(ts_img)
    # print(out.shape)
    
    out = non_max_suppression(out, conf_thres=0.25, iou_thres=iou_thres, multi_label=True)

    isNdl = False
    
    # Create image that includes bounding boxes and labels
    output_image = img
    for si, pred in enumerate(out):
        print(pred)
        if pred.size(dim=0) > 0:
            isNdl = True
    
    # Save it in a folder within the server
    cv2.imwrite(f'images_uploaded/{filename}', output_image)
    
    file_image = open(f'images_uploaded/{filename}', mode="rb")
    
    return "nine_dash_line" if isNdl else "binh_thuong"


@app.post("/predict/video") 
def predict_video(file: UploadFile = File(...)):

    # 1. VALIDATE INPUT FILE
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("mp4")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    
    this_dir = 'video_saved/{}'.format(filename.split(".")[0])
    if not os.path.exists(this_dir):
        os.mkdir(this_dir)
        with open('video_saved/{}'.format(filename), "wb+") as file_object:
            file_object.write(file.file.read())
            
        vidcap = cv2.VideoCapture('video_saved/{}'.format(filename))
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite("{}/frame{}.jpg".format(this_dir, count), image)     # save frame as JPEG file      
            success,image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1
    else:
        pass
    
    dataset = LoadImages(this_dir, img_size=640, stride=stride)
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # print(det, det.size())
            
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = '{}/{}'.format(save_dir, p.name)  # img.jpg
            txt_path = '{}/{}'.format(save_dir, p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Save results (image with detections)
            if save_img:
                # if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")
            if det.size(dim=0) > 0:
                return "nine_dash_line"
        
    return "binh_thuong"

# uvicorn main:app --host=127.0.0.1 --port=8000 --log-level=debug --reload