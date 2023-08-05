from modeling.processors import ParallelProcessor

import logging

class Runner():
    
    """
    The Runner gives tasks from an Order to a Processor.
    
    ...
    Attributes
    ----------
    order: Order
    
    Methods
    
    
    """
    
    def __init__(self, processor = None):
        if processor is None:
            self._processor = ParallelProcessor()
        else:
            self._processor = processor 
    
    def run(self, order):
        logging.info(f'Order {order.order_id} - started.')
        self._processor.process_in_background(order = order)
    