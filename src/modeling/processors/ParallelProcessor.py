from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import logging
import time
import threading

import sys

from .Processor import Processor

class ParallelProcessor(Processor):
    
    """
    The Runner gives tasks from an Order to a Processor.
    
    ...
    Attributes
    ----------
    order: Order
    
    Methods
    
    
    """
    
    def __init__(self, workers = 4, wait_time = 10):
        self._manager = Manager()
        self._max_workers = workers
        self._executor = ProcessPoolExecutor(max_workers = self._max_workers)
        self._wait_time = wait_time
        self._futures = []
    
    
    def process(self, order, output_queue):
        
        input_queue = self._manager.Queue() 
        for task in order.get_tasks():
            input_queue.put(task)
        
        self._futures.extend([self._executor.submit(_parallel_processor_wrapper,
                                                    df = order.df,
                                                    func = order.func,
                                                    queue_in = input_queue,
                                                    queue_out = output_queue)
                               for i in range(min(len(order.get_tasks()),
                                                  self._max_workers))]
                            )

            
    def process_in_background(self, order):

        output_queue = self._manager.Queue() 
        
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
            for future in self._futures:
                if future.exception() is not None:
                    sys.exit(future.exception())
            time.sleep(self._wait_time)
        logging.info(f'Order {order.order_id} - finished.')
        
        return 0
    
         
def _parallel_processor_wrapper(df, func, queue_in, queue_out):
    while not queue_in.empty():
        params = queue_in.get()
        ret = func(df = df,
                   params = params)
        queue_out.put(ret)