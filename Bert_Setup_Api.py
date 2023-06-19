
import os
import pandas as pd
from fastapi import FastAPI,Request
import uvicorn
import json
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import requests
#helper file
from src.database import ElasticTransformers
# from src.database_local_connection import ElasticTransformers
import src.custom as custfnc



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

app = FastAPI()
@app.post('/createIndex')
async def createIndex(request:Request):
    request.headers['Content-Type'] == 'application/json'
    userinput = await request.json() 
    DATA=userinput.get("data")
    FILENAME= userinput.get("filename")
    #cheking for blank value
    if DATA=="" or DATA is None:
        return {"Error: Please Provide Data (data parameter missing)"}
    if FILENAME=="" or FILENAME is None:
        return {"Error: Please Provide Filename (filename parameter missing)"}
    FILENAME=str(FILENAME).rstrip("pdf")+"csv"
    ##BERT Logic Begins
    csv_file=custfnc.texToCSV(DATA,FILENAME)#csv filename with path   
    csv_file_name = os.path.basename(csv_file) #csv filename
    INDEX_NAME=indexName(csv_file_name) #name of index
    et=ElasticTransformers(index_name=INDEX_NAME)
    if not et.ping():
        return {f"Error: Cannot connect to Elasticsearch index name: {INDEX_NAME}"}
    #creating Index Specification
    et.create_index_spec(
        text_fields=['index','data'],
        dense_fields=['data_embedding'],
        dense_fields_dim=768
    )
    #creating index
    et.create_index(index_name=INDEX_NAME)
    #Writing Index
    et.write_large_csv(
        csv_file,
        chunksize=10,
        embedder=embed_wrapper,
        field_to_embed='data'
    )
           
    return {f"Sucess: {INDEX_NAME} created on Elascticsearch Server"}
    
@app.post('/searchBot')
async def searchQuery(request:Request):
    print(type(request))
    request.headers['Content-Type'] == 'application/json'
    userinput = await request.json() 
    QUERY=userinput.get("query")
    #cheking for blank value
    if QUERY=="" or QUERY is None:
        return {"Error: Please Provide QUERY (query parameter missing)"}
    INDEX_NAME="chatbot_be_*"
    et=ElasticTransformers(index_name=INDEX_NAME)
    ESQUERY = str(custfnc.removeStopWord(QUERY))
    if not et.ping():
        return {f"Error: Cannot connect to Elasticsearch index name: {INDEX_NAME}"}
    df=et.search(ESQUERY,'data',type='dense',embedder=embed_wrapper,size=6)    
    if df.empty:
        df=et.search(ESQUERY,'data',type='match',embedder=embed_wrapper,size=6)    
        print("After Match Search")
    if df.empty:
        return []   
    df.to_csv("del.csv") 
    scaler = MinMaxScaler()
    df['_score_scale'] = scaler.fit_transform(df['_score'].values.reshape(-1,1))
    #et.closeconn()    
    myAnswers=[]    
    for rows in df.iterrows():
        temp={}
        answer= pd.DataFrame(rows[1]).iloc[1,0]   
        print(answer)         
        response = requests.post('http://127.0.0.1:5009/answer', json={
            "question":QUERY,
            "passage":answer
        })     
        filelink=custfnc.getFilelink(answer)  
        temp['answer']=response.text.replace("\"","")
        temp['fileLink']=filelink
        temp['fileName']=filelink.split("/")[-1]
        temp['pageNumber']=custfnc.getPageNum(answer)
        myAnswers.append(temp)
         
    return myAnswers
    
    

    
    

if __name__ == '__main__':
   uvicorn.run("Bert_Setup_Api:app",port=5010,reload=True)


