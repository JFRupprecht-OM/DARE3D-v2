import numpy as np
import torch

class MonaiAugmentationComponent:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data_dict):
        data_dict = self.transform(data_dict)
        return data_dict

class MonaiAugmentationWrapper:
    def __init__(self, components, device="gpu", prob=0.5):
        self.augmentations = components
        self.device = torch.device("cuda:0") if device=="gpu" else torch.device("cpu")
        self.prob = prob
            
    def __call__(self, data_dict):
        res_data_dict = {}
        for name, array in data_dict.items():
            value = array
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(array, device=self.device)
            res_data_dict.update({name: value})
        
        # Probability to augment a sample with the given
        # transform pipeline
        if np.random.rand() <= self.prob:
            for _, aug in self.augmentations.items():
                res_data_dict = aug(res_data_dict)
        return res_data_dict