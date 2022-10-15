from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import time


#- See https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor
#- Under construction
#- Currently unable to pickle the function I think 
class Parallel_Processor:
    
    """
    
    """
    
    def __init__(self, df, func, workers):
        self._manager = Manager()
        self._func = func
        self._max_workers = workers
    
    def get_new_queue(self):
        return self._manager.Queue() 
    
    def _wrapper_func(self, queue_in, queue_out):
        while not queue_in.empty():
            ret = self._func(df = self.df, params = queue_in.get())
            queue_out.put(ret)
    
    def process(self, queue_in, queue_out):
        with ProcessPoolExecutor(max_workers = self._max_workers) as executor:
            self.futures = [executor.submit(self._wrapper_func,
                                            queue_in = queue_in,
                                            queue_out = queue_out) for i in range(self._max_workers)]
        
    def is_still_processing(self):
        #-TODO: Figure out a use for this
        return any([future.running() for future in self.futures])
    