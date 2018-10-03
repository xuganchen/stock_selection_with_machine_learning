from enum import Enum
from abc import ABCMeta, abstractmethod
from sklearn.metrics import roc_auc_score
import numpy as np

class ModelType(Enum):
    LR = 0
    RF = 1
    SVM = 2
    DNN = 3
    LSTM = 4
    NB = 5
    GBM = 6
    BAG = 7
    ET = 8
    ADA = 9
    EXS = 10


class AbstractModel(object):
    '''
    Abstract class of model
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def _generate_model(self):
        '''
        Generate model with empty class
        '''
        raise NotImplementedError("Should implement _generate_model()")

    @abstractmethod
    def evalution(self, is_GA = False):
        '''
        Evalution model with data 'self.X_test' and 'self.Y_test'

        :param is_GA: True or False. Whether using by Genetic Algorithm
        :return:
            if is_GA is True: AUC
            if is_GA is False: Accuracy, Precision, Recall, F1, TPR, FPR, AUC
        '''
        raise NotImplementedError("Should implement evalution()")

    @abstractmethod
    def save_model(self, fpath):
        '''
        Save model into the dictionary 'fpath'.
        The model will be saved into the 'fpath' in format:
            if keras: '_model_architecture.json' and '_model_weights.h5'
            if sklearn: '_model.pkl'

        :param fpath: the dictionary of saving-model
        :return:
        '''
        raise NotImplementedError("Should implement save_model()")

    @abstractmethod
    def _load_model(self, fpath):
        '''
        Loading model from the dictionary 'fpath'.
        The model will be loaded from the 'fpath' in format:
            if keras: '_model_architecture.json' and '_model_weights.h5'
            if sklearn: '_model.pkl'

        :param fpath: the dictionary of saving-model
        :return:
        '''
        raise NotImplementedError("Should implement _load_model()")

    @abstractmethod
    def evalution_with_data(self, X_test, Y_test):
        '''
        Evalution model with data 'X_test' and 'Y_test'

        :param X_test: the data using into model.predict
        :param Y_test: the true target data
        :return: Accuracy, Precision, Recall, F1, TPR, FPR, AUC
        '''
        raise NotImplementedError("Should implement evalution_with_data()")

    @abstractmethod
    def predict(self, X_test):
        '''
        Predict data 'X_test' and return the classified output

        :param X_test: the data using into model.predict
        :return: np.array of classifed data(0 or 1)
        '''
        raise NotImplementedError("Should implement predict()")

    @abstractmethod
    def predict_proba(self, X_test):
        '''
        Predict data 'X_test' and return the probability output

        :param X_test: the data using into model.predict_proba
        :return: np.array of probability data([0, 1])
        '''
        raise NotImplementedError("Should implement predict_proba()")


    def calculate(self, Y_test, Y_pred, is_GA = False):
        '''

        :param Y_test: the true target data
        :param Y_pred: the predicted target data
        :param is_GA: True or False. Whether using by Genetic Algorithm
        :return:
            if is_GA is True: AUC
            if is_GA is False: Accuracy, Precision, Recall, F1, TPR, FPR, AUC
        '''
        if is_GA:
            AUC = roc_auc_score(Y_test, Y_pred)
            return AUC
        else:
            Y_pred_int = np.round(Y_pred)

            TP, FP, FN, TN = 0, 0, 0, 0
            for i in range(len(Y_pred_int)):
                if Y_pred_int[i] == Y_test[i]:
                    if Y_pred_int[i] == 1:
                        TP += 1
                    else:
                        TN += 1
                else:
                    if Y_pred_int[i] == 1:
                        FP += 1
                    else:
                        FN += 1

            Accuracy = (TP + TN) / (TP + TN + FP + FN)
            Precision = TP / (TP + FP)
            Recall = TP / (TP + FN)
            F1 = 2 * Precision * Recall / (Precision + Recall)
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            AUC = roc_auc_score(Y_test, Y_pred)
            return Accuracy, Precision, Recall, F1, TPR, FPR, AUC