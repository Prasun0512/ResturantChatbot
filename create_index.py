from elasticsearch import Elasticsearch
import os

# client = Elasticsearch(hosts="http://localhost:9200")
# fl = os.listdir("CSV")
# for file in fl: 
#     name= file.rstrip(".pdf.csv")  
#     client.indices.create(index="chatbot_"+str.lower(name), body={
#    'settings' : {
#          'index' : {
#               'number_of_shards':3
#          }
#    }
# })


def indexName(name):
    iname= name.rstrip(".pdf.csv")
    iname= "chatbot_"+str.lower(iname)
    return name