from queue import Queue

class Serial_Processor:
    
    """
    
    
    """
    
    def __init__(self, df, func):
        self.df = df
        self.func = func
        
    def get_new_queue(self):
        return Queue() 
        
    def process(self, queue_in, queue_out):
        
        #- TODO: 
        #- 1. Any resampling 
        
        while not queue_in.empty():
            params = queue_in.get()
            result = self.func(df = self.df, params = params)
            queue_out.put(result)