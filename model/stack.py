from enum import Enum
from abc import ABCMeta, abstractmethod


class ModelType(Enum):
    LR = 0
    RF = 1
    SVM = 2
    DNN = 3
    LSTM = 4
    STACK = 5



class AbstractModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def evalution(self):
        raise NotImplementedError("Should implement evalution()")