from augmend import Augmend
import numpy as np

class AugmendComponentWrapper:
    def __init__(self, input, target, probability):
        self.input = input
        self.target = target
        self.probability = probability


class AugmendWrapper:
    def __init__(self, components):
        self.aug = Augmend()
        for augmentation_name, augmentation in components.items():
            self.aug.add(
                [
                    augmentation.input,
                    augmentation.target,
                ],
                probability=augmentation.probability,
            )

    def __call__(self, x, y):
        # From (T, X, Y, Z) to (X, Y, Z, T) for augmentations
        x = np.moveaxis(x, 0, -1)
        x, y = self.aug([x, y])
        x = np.moveaxis(x, -1, 0)
        return x, y