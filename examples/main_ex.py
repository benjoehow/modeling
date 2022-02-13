import pandas as pd
import json

from threading import Thread

from modeling import expediator 
from modeling.processors import serial_processor

@click.command()
@click.option('--train_data_filepath', help = 'The filepath of the training data.')
@click.option('--config_filepath', help = 'The filepath of the configuraiton file.')
@click.option('--output_filepath', help = 'The filepath to save results to.')
def main(train_data_location, config_location):
    
    train_data = pd.read_csv(train_data_location)
    
    with open(config_location, 'r') as config_file:
        config = json.load(config_file)
        
    execute_training(train_data = train_data,
                     config = config,
                     output_filepath = output_filepath)
    

def execute_training(train_data, config, output_filepath):
    
    exptr = expediator(df = train_data,
                       config = config)
    
    processor = serial_processor(df = train_data,
                                 func = exptr.get_train_func())
    
    input_queue = processor.get_new_queue()
    runner.load_input_queue(input_queue)
    output_queue = processor.get_new_queue()
    
    background_process = Thread(target = processor.process,
                                kwargs = {"queue_in": input_queue, "queue_out": output_queue})
    background_process.start()
    
    while not input_queue.empty() and not output_queue.empty():
        if not output_queue.empty():
            runner.log_completed_task(output = output_queue.get())
    

if __name__ == "__main___":
    main()
    