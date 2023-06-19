from allennlp.predictors.predictor import Predictor
import allennlp_models.rc
from flask import Flask, jsonify, request
import json

app = Flask(__name__)

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/transformer-qa.2021-02-11.tar.gz")

@app.route('/answer', methods=['POST'])
def process():
    request.headers['Content-Type'] == 'application/json'

    json_inp = request.json   
    question = json_inp["question"]
    passage = json_inp["passage"]
    

    result = predictor.predict(passage=passage, question=question)


    answer = result['best_span_str']
    score = result['best_span_probs']  
  
    return jsonify(answer=answer,score=score)

if __name__ == "__main__":
    # app.config['DEBUG'] = DEBUG_ENABLED
    app.run(host='0.0.0.0',port=8080)



