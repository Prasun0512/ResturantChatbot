
import os
import pandas as pd
from fastapi import FastAPI,Request
import requests
import uvicorn
from src.database import ElasticTransformers
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from tqdm import trange
#helper file
import src.custom as txtcsv




bert_embedder = SentenceTransformer('bert-base-nli-mean-tokens')


def embed_wrapper(ls):
    """
    Helper function which simplifies the embedding call and helps lading data into elastic easier
    """
    results=bert_embedder.encode(ls, convert_to_tensor=True)
    results = [r.tolist() for r in results]
    return results

def indexName(filename):
    return str.lower("chatbot_be_"+filename.rstrip(".csv_"))

def select_search_results(df,top_n=10):
    # four tokens or more (filtering out some meaningless headlines)
    df=df[df.data.apply(lambda x: len(x.split())>4)].copy()
    # remove exact duplicates
    df=df.groupby('data', as_index=False).first()
    df=df.sort_values('_score',ascending=False)
    df=df.reset_index(drop=True)
    return df.head(top_n)

INDEX_NAME="chatbot_be_test"
def search(QUERY):
    print("search called")
    et=ElasticTransformers(index_name=INDEX_NAME)
    if not et.ping():
        return {f"Error: Cannot connect to Elasticsearch index name: {INDEX_NAME}"}
    else:
        print("Connected")
    
    df=et.search(QUERY,'data',type='dense',embedder=embed_wrapper,size=6)
    print("After Dense Search")
    if df.empty:
        df=et.search(QUERY,'data',type='match',embedder=embed_wrapper,size=6)    
        print("After Match Search")
    scaler = MinMaxScaler()
    df['_score_scale'] = scaler.fit_transform(df['_score'].values.reshape(-1,1))
    et.closeconn()
    print("Before For")
    for rows in df.iterrows():
        answer= pd.DataFrame(rows[1]).iloc[1,0]
        query = {'question':QUERY, 'passage':answer}
        print(query)
        response = requests.post('http://127.0.0.1:5008/answer', json=query)
        print("*****",response.json())
    
   
    

search("what is Artificial Intelligence")