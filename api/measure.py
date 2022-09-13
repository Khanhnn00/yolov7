import sys
sys.path.insert(1, '/opt/work/yolov7')

import os
import torch
import random
from models.experimental import attempt_load
from tqdm import tqdm

from api_utils import predict_video_path
import config as cf

def init_model(cf):

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    
    model = attempt_load(cf.model['weight'], map_location=device)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    stride = int(model.stride.max())  # model stride
    imgsz = cf.model['image_size']
    model.eval()

    save_dir = 'results_detect'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('Finished init model')

    return model, names, colors, imgsz, stride, save_dir, device

# def run(url):
    
#     label = predict_video_path(url, cf)
#     return label
    # label = 'binh_thuong' if label['label'] == True else 'hsts_invalid'
    # print(label)
    
if __name__ == '__main__':
    import config as cf
    # url = './video_temp/1FtLrM_dyZc.mp4'
    # run(url)
    model, names, colors, imgsz, stride, save_dir, device = init_model(cf)
    pth = '/opt/work/dataset/downloaded'
    videos = os.listdir(pth)
    save_dir = 'exp_ndl_frame5_margin'
    cnt = 0
    total = len(videos)
    print(total)
    sum_hard = 0
    sum_soft = 0
    ndl = normal = lst_soft = lst_hard = []
    for vid in tqdm(videos):
        url = '{}/{}'.format(pth, vid)
        res, rating_hard, rating_soft, vid_name  = predict_video_path(url, model, stride, device, cf, save_dir)
        print(res)
        if res == "ndl":
            print(url)
            cnt += 1
            ndl.append(vid_name)
        else:
            normal.append(vid_name)

        if rating_hard < 0.5:
            lst_hard.append(vid_name)
        
        if rating_soft > 0.5:
            lst_soft.append(vid_name)
        sum_hard += rating_hard
        sum_soft += rating_soft
    
    print(cnt, total)
    # print(lst_hard,)
    # print(lst_soft)
    print('Avg hard margin: {}'.format(sum_hard/total))
    print('Avg soft margin: {}'.format(sum_soft/total))
    # print(ndl)
    # print(normal)