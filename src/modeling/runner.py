from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import time
import threading

import sys


class Runner():
    
    """
    The Runner gives tasks from an Order to a Processor.
    
    ...
    Attributes
    ----------
    order: Order
    
    Methods
    
    
    """
    
    def __init__(self, workers = 1, wait_time = 10):
        self._manager = Manager()
        self._max_workers = workers
        self._executor = ProcessPoolExecutor(max_workers = self._max_workers)
        self._wait_time = wait_time
        self._futures = []
    
    def run(self, order):
        print("process started")
        self._process_in_background(order = order)
    
    def _process(self, order, output_queue):
        
        input_queue = self._manager.Queue() 
        for task in order.get_tasks():
            input_queue.put(task)
        
        self._futures.extend([self._executor.submit(_parallel_processor_wrapper,
                                              df = order.df,
                                              func = order.func,
                                              queue_in = input_queue,
                                              queue_out = output_queue)
                              for i in range(self._max_workers)]
                            )

            
    def _process_in_background(self, order):

        output_queue = self._manager.Queue() 
        
        process_thread = threading.Thread(target = self._process,
                                          name = "processor",
                                          kwargs = {"order": order,
                                                    "output_queue": output_queue}
                                         )
        process_thread.start()
        
        while(not order.is_finished): 
            while not output_queue.empty():
                result = output_queue.get()
                print("adding result")
                order.add_result(result = result)
            for future in self._futures:
                if future.exception() is not None:
                    sys.exit(future.exception())
            time.sleep(self._wait_time)
        print("order finished")
        
        return 0
    
         
def _parallel_processor_wrapper(df, func, queue_in, queue_out):
    while not queue_in.empty():
        params = queue_in.get()
        ret = func(df = df,
                   params = params)
        queue_out.put(ret)