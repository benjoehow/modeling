import threading
import time

class Runner():
    
    def __init__(self, order):
        self.order = order
    
    def run(self, processor):
        output_queue = self._process_in_background(processor = processor)
        while(not self.order.is_finished):
            self._empty_output_queue(output_queue = output_queue)
            time.sleep(10)

    def _empty_output_queue(self, output_queue):
        while not output_queue.empty():
            result = output_queue.get()
            self.order.add_result(result = result)
            
    def _process_in_background(self, processor):
        input_queue = processor.get_new_queue()
        self._load_input_queue(input_queue = input_queue)
        
        output_queue = processor.get_new_queue()
        
        process_thread = threading.Thread(target = processor.process,
                                          name = "processor",
                                          kwargs = {"queue_in": input_queue,
                                                    "queue_out": output_queue}
                                         )
        process_thread.start()
        return output_queue
    
    def _load_input_queue(self, input_queue):
        for task in self.order.get_tasks():
            input_queue.put(task)
    
    def get_results(self):
        return self.order.get_results()