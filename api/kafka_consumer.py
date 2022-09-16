import os
import torch
import random
import time
from json import loads, dumps
import requests
from kafka import KafkaConsumer

import sys
# print(sys.path)

import inspect


from api_utils import predict_video_path
from models.experimental import attempt_load
import config as cf

def init_model(cf):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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

def run():
    consumer = KafkaConsumer(topics,
                            bootstrap_servers=bootstrap_servers,
                            auto_offset_reset='earliest',
                            enable_auto_commit=True,
                            group_id=group_id,
                            value_deserializer=lambda x: loads(x.decode('utf-8')))
    print("Message ")
    for message in consumer:
        print('message')
        print('running')
        # try:
        print ("%s:%d:%d: value=%s" % (message.topic, message.partition, message.offset, message.value))
        data = message.value
        url = data['url']
        create_time = data['create_time']
        label = data['nine_dash_line']

        print('finish getting info')

        if label != None:
            continue

        # process video
        # model, names, colors, imgsz, stride, save_dir, device = init_model(cf)
        label = predict_video_path(url, model, stride, device, cf, save_dir)
        print(label)
        # label = 'binh_thuong' if label['label'] == True else 'hsts_invalid'
        # print(label)
        
        record = {
            'url': url,
            'label': label,
            'model': 'nine_dash_line',
            'create_time': create_time
        }
        resp = requests.post(callback, headers = {"Content-Type": "application/json"}, data = dumps(record))
        consumer.commit_async()
        print("Update record: %s %s \n" % (resp.status_code, resp.text))
        # except Exception as error:
        #     print("Error: ", error)


if __name__ == '__main__':
    import config as cf
    
    print("Inint model \n")
    model, names, colors, imgsz, stride, save_dir, device = init_model(cf)
    # vis_processor = VideoInference(cf.model, cf.device)
    
    print("Init kafka \n")

    os.environ['KAFKA_SERVER'] = "10.8.5.83:9092,10.8.5.45:9092,10.8.6.193:9092"
    os.environ['KAFKA_TOPIC'] = "ml-video-storage-censorship-dev"
    os.environ['KAFKA_GROUP'] = "adtechHCM-ndl" 
    os.environ['KAFKA_CALLBACK'] = "http://172.18.5.44:8000/mlbigdata/cv/video-storage-dev/update_label"

    bootstrap_servers = os.environ.get('KAFKA_SERVER', None)
    topics = os.environ.get('KAFKA_TOPIC', None)
    group_id = os.environ.get('KAFKA_GROUP', None)
    callback = os.environ.get('KAFKA_CALLBACK', None)
    if bootstrap_servers and topics and group_id and callback:
        bootstrap_servers = [h.strip() for h in bootstrap_servers.split(',')]
        idle = 0
        while True:
            try:
                run()
                idle += 1
                print("Try: ", idle)
            except Exception as error:
                print(error)