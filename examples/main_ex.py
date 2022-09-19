import pandas as pd
import json

from threading import Thread

from modeling.processors import Serial_Processor
from modeling.processors import Serial_Processor


def main(df, config):
    
    train_data = pd.read_csv(train_data_location)
    
    with open(config_location, 'r') as config_file:
        config = json.load(config_file)
        
    execute_training(train_data = train_data,
                     config = config,
                     output_filepath = output_filepath)
    
    expeditor = Expeditor(config = config)
    order = expeditor.get_order(df = wine)
    runner.run(processor = sp)
    
    return 0
    

def execute_training(train_data, config, output_filepath):
    
    with open(config_file, 'r') as config_file:
        config = json.load(config_file)
    
    train_data = pd.read_csv(train_data_location)
    
    main(df = train_data
    

if __name__ == "__main___":
    main()
    