from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    def get_name(self) -> str:
        return self.__class__.__name__

    def get_grad_norm_weights(self):
        return self.parameters()
