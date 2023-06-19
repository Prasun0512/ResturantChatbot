import pandas as pd
import os
from database import ElasticTransformers
from elasticsearch import Elasticsearch
from create_index import indexName
from sentence_transformers import SentenceTransformer

count =1

bert_embedder = SentenceTransformer('bert-base-nli-mean-tokens')

fl = os.listdir("CSV")

def embed_wrapper(ls):
    """
    Helper function which simplifies the embedding call and helps lading data into elastic easier
    """
    results=bert_embedder.encode(ls, convert_to_tensor=True)
    results = [r.tolist() for r in results]
    return results

for file in fl:    
    if count==5:
        break
    filename=os.path.join("CSV",file)
    df=pd.read_csv(filename)
    INDEX_NAME=indexName(file)
    et=ElasticTransformers(index_name=INDEX_NAME)
    et.create_index_spec(
        text_fields=['index','data'],
        dense_fields=['data'],
        dense_fields_dim=768
    )
    print(indexName(file))
    et.write_large_csv(os.path.join("CSV",file),
                  chunksize=10,
                  embedder=embed_wrapper,
                  field_to_embed='data')
    
    es = Elasticsearch(hosts="http://localhost:9200")
    es.indices.stats(index=INDEX_NAME)
    et.write_large_csv(os.path.join("CSV",file),
                  chunksize=1000,
                  embedder=embed_wrapper,
                  field_to_embed='data')
    print("done")
    count= int(count)+1
