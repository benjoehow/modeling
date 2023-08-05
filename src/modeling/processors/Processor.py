from abc import ABC, abstractmethod

class Processor(ABC):
    
    """
    The AbstractProccessor ...
    
    ...
    Attributes
    ----------
    order: Order
    
    Methods
    
    
    """
    
    @abstractmethod
    def process(self, order, output_queue):
        pass

    @abstractmethod
    def process_in_background(self, order):
        pass
