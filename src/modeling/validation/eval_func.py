class EvalFuncWrapper(): 
    
    def __init__(self,
                 family_id,
                 metric_id, 
                 eval_func, 
                 tags = []):
        
        self.family_id = family_id
        self.id = metric_id
        self.eval_func = eval_func
        self.requires_binary_output = False
        self._compile_tags(tags)
        
    def _compile_tags(self, tags):
        if "requires_binary_output" in tags: 
            self.requires_binary_output = True
            
    def __call__(self, *args, **kwargs):    
        return self.eval_func(*args, **kwargs)
