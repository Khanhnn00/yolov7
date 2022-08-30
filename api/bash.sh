# export KAFKA_SERVER="10.8.5.83:9092,10.8.5.45:9092,10.8.6.193:9092" 
# KAFKA_TOPIC="ml-video-storage-censorship-dev"
# KAFKA_GROUP="adtechHCM" 
# KAFKA_CALLBACK="http://172.18.5.44:8000/mlbigdata/cv/video-storage-dev/update_label"
python api/kafka_consumer.py