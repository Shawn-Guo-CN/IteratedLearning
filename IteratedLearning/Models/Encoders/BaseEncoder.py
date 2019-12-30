from torch.nn import module


class BaseEncoder(module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        raise NotImplementedError

