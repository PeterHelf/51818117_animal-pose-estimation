"""Ways to initialize weights for PyTorch modules"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn

class BaseWeightInitializer(ABC):
    """Class to used to initialize model weights"""

    @abstractmethod
    def init_weights(self, model: nn.Module) -> None:
        """Initializes weights for a model.

        Args:
            model: The model for which to initialize weights
        """

class Normal(BaseWeightInitializer):
    """Class to used to initialize model weights using a normal distribution

    Weights are initialized with a normal distribution, and biases are initialized to 0.

    Attributes:
        std: the standard deviation to use to initialize weights
    """

    def __init__(self, std: float = 0.001):
        self.std = std

    def init_weights(self, model: nn.Module) -> None:
        for name, module in model.named_parameters():
            if "bias" in name:
                nn.init.constant_(module, 0)
            else:
                nn.init.normal_(module, std=self.std)

class Dekr(BaseWeightInitializer):
    """Class to used to initialize model weights in the same way as DEKR

    Attributes:
        std: the standard deviation to use to initialize weights
    """

    def __init__(self, std: float = 0.001):
        self.std = std

    def init_weights(self, model: nn.Module) -> None:
        for name, module in model.named_parameters():
            if "bias" in name:
                nn.init.constant_(module, 0)
            else:
                nn.init.normal_(module, std=self.std)

            if hasattr(module, "transform_matrix_conv"):
                nn.init.constant_(module.transform_matrix_conv.weight, 0)
                if hasattr(module, "bias"):
                    nn.init.constant_(module.transform_matrix_conv.bias, 0)
            if hasattr(module, "translation_conv"):
                nn.init.constant_(module.translation_conv.weight, 0)
                if hasattr(module, "bias"):
                    nn.init.constant_(module.translation_conv.bias, 0)

class Rtmpose(BaseWeightInitializer):
    """Class to used to initialize head weights in the same way as RTMPose"""

    def init_weights(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 1)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)
