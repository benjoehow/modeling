from queue import Queue

import logging
import time
import threading

import sys

from .Processor import Processor


class SequentialProcessor(Processor):
    
    """
    The Runner gives tasks from an Order to a Processor.
    
    ...
    Attributes
    ----------
    order: Order
    
    Methods
    
    
    """
    
    def __init__(self, wait_time = 10):
        self._wait_time = wait_time
    
    def process(self, order, output_queue):
        
        for params in order.get_tasks():
            #-TODO: See if the task has been completed yet
            ret = order.func(df = order.df, params = params)
            output_queue.put(ret)

            
    def process_in_background(self, order):

        output_queue = Queue() 
        
        process_thread = threading.Thread(target = self.process,
                                          name = "processor",
                                          kwargs = {"order": order,
                                                    "output_queue": output_queue}
                                         )
        process_thread.start()
        
        while(not order.is_finished): 
            while not output_queue.empty():
                result = output_queue.get()
                order.add_result(result = result)
            time.sleep(self._wait_time)
        logging.info(f'Order {order.order_id} - finished.')