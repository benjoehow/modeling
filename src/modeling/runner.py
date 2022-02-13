import pandas as pd

class Runner:
    
    def __init__(self, tasks):
        self.tasks = tasks
        self.completed = pd.DataFrame()
        #self.func = func
        
    def load_input_queue(queue):
        for task in tasks:
            queue.put(task)
        
    def log_completed_task(task):
        self.completed.append(task)