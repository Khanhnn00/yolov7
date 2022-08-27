import os
import time
from json import loads, dumps
import requests
from kafka import KafkaConsumer

from utils import predict_video
import config as cf

def run():
    consumer = KafkaConsumer(topics,
                            bootstrap_servers=bootstrap_servers,
                            auto_offset_reset='earliest',
                            enable_auto_commit=True,
                            group_id=group_id,
                            value_deserializer=lambda x: loads(x.decode('utf-8')))
    print("Message ")
    for message in consumer:
        try:
            print ("%s:%d:%d: value=%s" % (message.topic, message.partition, message.offset, message.value))
            data = message.value
            url = data['url']
            create_time = data['create_time']
            label = data['hs_ts']
            if label != None:
                continue

            # process video
            label = predict_video(url, cf)
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
        except Exception as error:
            print("Error: ", error)


if __name__ == '__main__':
    import config as cf
    
    print("Inint model \n")
    vis_processor = VideoInference(cf.model, cf.device)
    
    print("Init kafka \n")
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