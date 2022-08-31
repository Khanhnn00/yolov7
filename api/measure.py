import os
import time
from json import loads, dumps
import requests

import sys
# print(sys.path)

import inspect


from api_utils import predict_video_path
import config as cf

def run(url):
    
    label = predict_video_path(url, cf)
    print(label)
    # label = 'binh_thuong' if label['label'] == True else 'hsts_invalid'
    # print(label)
    
if __name__ == '__main__':
    import config as cf
    url = './video_temp/1FtLrM_dyZc.mp4'
    run(url)