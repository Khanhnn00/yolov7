import requests
import time
from datetime import datetime
import os
import urllib3
import torch
from pathlib import Path
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from ..models.experimental import attempt_load
from ..utils.datasets import LoadImages, letterbox
from ..utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh
from ..utils.plots import plot_one_box
from ..utils.torch_utils import time_synchronized

def check_valid_video_folder():
    if not os.path.exists("video_temp"):
        os.makedirs("video_temp")

def download_video(video_url):
    # header = random_headers()
    # response = requests.get(video_url, headers=header, stream=True, verify=False, timeout=5)
    try:
        response = requests.get(video_url, verify=False)
        # prefix = "video_temp/" + datetime.today().strftime("%Y_%m_%d_%H_%M_%S_")
        prefix = "video_temp/"
        vid_name = prefix + video_url.split('/')[-1]
        # print("vid_name: ", vid_name)
        with open(vid_name, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        print("download_video", e)
        vid_name = None
    return vid_name

def init_model(cf):

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    half = device != 'cpu'
    model = attempt_load(cf.model['weight'], map_location=device)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    stride = int(model.stride.max())  # model stride
    imgsz = cf.model['image_size']
    model.eval()

    save_dir = 'results_detect'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    return model, names, colors, imgsz, stride, save_dir

def predict_video(video_url, cf):

    video_name = download_video(video_url)
    
    this_dir = 'video_temp/{}'.format(video_name.split("/")[-1].split(".")[0])
    if not os.path.exists(this_dir):
        os.mkdir(this_dir)
            
        vidcap = cv2.VideoCapture(video_name)
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

    model, names, colors, imgsz, stride, save_dir = init_model(cf)
    
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
        pred = non_max_suppression(pred, cf.model['conf_thres'], cf.model['iou_thres'])
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
                    if cf.save_img:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if cf.save_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Save results (image with detections)
            if cf.save_img:
                # if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")
            if det.size(dim=0) > 0:
                return "nine_dash_line"
        
    return "binh_thuong"