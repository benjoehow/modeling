import os
import sys
import pandas as pd

import threading 
# Flask
from flask import Flask, request
from gevent.pywsgi import WSGIServer

from modeling import get_order, Runner

# Declare a flask app
app = Flask(__name__)

runner = Runner(workers = 4)
train_orders = {}


@app.route('/', methods=['GET'])
def index():
    # Main page
    return {'test': 0}

@app.route('/jobs', methods=['GET'])
def get_train_jobs():
    return ', '.join(list(train_orders.keys()))


@app.route('/train/<job_id>', methods=['POST', 'GET'])
def train(job_id):
    
    if request.method == 'POST':
        json_data = request.get_json()

        config = json_data['config']
        df = pd.DataFrame.from_dict(json_data['df'])

        order = get_order(df = df, config = config)
        train_orders[job_id] = order

        process_thread = threading.Thread(target = runner.run,
                                          name = "train_" + job_id,
                                          kwargs = {"order": order}
                                          )
        process_thread.start()
        
        return job_id
        
    if request.method == 'GET':
        if job_id in train_orders:
            if train_orders[job_id].is_finished:
                return train_orders[job_id].get_results()
            else: 
                return {'status': 'still_training'}
    


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5050), app)
    http_server.serve_forever()