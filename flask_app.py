import logging
import os
import pandas as pd
import sys
import uuid

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

@app.route('/train/jobs', methods=['GET'])
def get_train_jobs():
    return ', '.join(list(train_orders.keys()))


@app.route('/train/', methods=['POST'])
def train():
    
    json_data = request.get_json()

    config = json_data['config']
    df = pd.DataFrame.from_dict(json_data['df'])

    order = get_order(df = df, config = config)
    job_id = order.order_id
    
    logging.info(f'New train order ({job_id}) received.')
    
    train_orders[job_id] = order

    process_thread = threading.Thread(target = runner.run,
                                      name = "train_" + job_id,
                                      kwargs = {"order": order}
                                    )
    process_thread.start()
    
    return job_id

@app.route('/train/status/<job_id>', methods=['GET'])
def train_status(job_id):
    
    if job_id in train_orders:
        if train_orders[job_id].is_finished:
            return train_orders[job_id].get_results()
        else:
            return f'{train_orders[job_id].completed_task_count} / {train_orders[job_id].total_tasks} tasks completed.'
    


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)
    logging.basicConfig(filename='flask_app.log', filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    
    http_server = WSGIServer(('0.0.0.0', 5051), app)
    http_server.serve_forever()