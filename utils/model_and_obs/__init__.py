

class BasePolicyCustomManager:
    def __init__(self, logdir, device):
        self.handler = None
        self.logdir = logdir
        self.device = device
        
    def set_handler(self, handler):
        self.handler = handler
    
    def get_action(self):
        raise NotImplementedError("Subclasses must implement this method")
    