import pandas as pd
from fastapi import FastAPI, Request
from src.database import ElasticTransformers
from sklearn.preprocessing import MinMaxScaler
from typing import List
import aiohttp
from sentence_transformers import SentenceTransformer
import src.custom as custfnc

app = FastAPI()

bert_embedder = SentenceTransformer('bert-base-nli-mean-tokens')

def embed_wrapper(ls):
    """
    Helper function which simplifies the embedding call and helps lading data into elastic easier
    """
    results=bert_embedder.encode(ls, convert_to_tensor=True)
    results = [r.tolist() for r in results]
    return results

# Initialize ElasticSearch client and scaler
INDEX_NAME = "chatbot_be_*"
et = ElasticTransformers(index_name=INDEX_NAME)
scaler = MinMaxScaler()

async def answer_question(QUERY, answer):
    async with aiohttp.ClientSession() as session:
        response = await session.post('http://127.0.0.1:5009/answer', json={
            "question": QUERY,
            "passage": answer
        })
        return response.text.replace("\"","")

@app.post('/searchBot')
async def search_query(request: Request):
    try:
        user_input = await request.json()
        QUERY = user_input.get("query")
        QUERY = " " + QUERY + " "
        
        # Check for blank value
        if not QUERY:
            return {"Error: Please Provide QUERY (query parameter missing)"}
        
        ESQUERY = str(custfnc.removeStopWord(QUERY))
        
        if not et.ping():
            return {f"Error: Cannot connect to Elasticsearch index name: {INDEX_NAME}"}
        
        df = et.search(ESQUERY, 'data', type='dense', embedder=embed_wrapper, size=6)
        if df.empty:
            df = et.search(ESQUERY, 'data', type='match', embedder=embed_wrapper, size=6)
        
        if df.empty:
            return []
        
        df['_score_scale'] = scaler.fit_transform(df['_score'].values.reshape(-1, 1))
        
        # Use apply method to apply the function to all rows of the dataframe
        df['answer'] = df['answer'].apply(lambda answer: answer_question(QUERY, answer))
        
        df['fileLink'] = df['answer'].apply(lambda answer: custfnc.getFilelink(answer))
        df['fileName'] = df['fileLink'].apply(lambda link: link.split("/")[-1])
        df['pageNumber'] = df['answer'].apply(lambda answer: custfnc.getPageNum(answer))
        
        return df[['answer', 'fileLink', 'fileName', 'pageNumber']].to_dict('records')
    
    except Exception as e:
        return {"Error": str(e)}

if __name__ == '__main__':
   uvicorn.run("Search:app",port=5008,reload=True)