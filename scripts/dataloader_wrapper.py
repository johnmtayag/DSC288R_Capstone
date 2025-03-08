import torch

### 
class DataloaderWrapper:
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                                     batch_size=batch_size, 
                                                     shuffle=shuffle, 
                                                     num_workers=num_workers)