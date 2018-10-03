from .base import ModelType, AbstractModel

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import os
import numpy as np

class RandomForest(AbstractModel):
    def __init__(self,
                 X_train = None,
                 Y_train = None,
                 X_test = None,
                 Y_test = None,
                 fpath = None):
        '''
        The RandomForest model class

        :param X_train: training data
        :param Y_train: training target
        :param X_test: testing data
        :param Y_test: testing target
        :param fpath: the dictionart path of loading-model
        '''
        if fpath is None and (
                (X_train is None) or
                (Y_train is None) or
                (X_test is None) or
                (Y_test is None)
        ):
            raise ValueError("Should input 'X_train, Y_train, X_test, Y_test' or 'fpath'")

        self.type = ModelType.RF

        if fpath is None:
            self.X_train = X_train
            self.Y_train = Y_train
            self.X_test = X_test
            self.Y_test = Y_test
            self.model = self._generate_model()
        else:
            self.model = self._load_model(fpath)

    def _generate_model(self):
        '''
        Generate model with empty class
        '''
        model = RandomForestClassifier(n_estimators=100, max_depth=4, verbose=0)
        return model

    def evalution(self, is_GA = False):
        '''
        Evalution model with data 'self.X_test' and 'self.Y_test'

        :param is_GA: True or False. Whether using by Genetic Algorithm
        :return:
            if is_GA is True: AUC
            if is_GA is False: Accuracy, Precision, Recall, F1, TPR, FPR, AUC
        '''
        self.model.fit(self.X_train, self.Y_train)
        Y_pred = self.model.predict_proba(self.X_test)
        Y_pred = Y_pred[:,1]
        Y_test = self.Y_test
        return self.calculate(Y_test, Y_pred, is_GA=is_GA)

    def _load_model(self, fpath):
        '''
        Loading model from the dictionary 'fpath'.
        The model will be loaded from the 'fpath' in format:
            if keras: '_model_architecture.json' and '_model_weights.h5'
            if sklearn: '_model.pkl'

        :param fpath: the dictionary of saving-model
        :return:
        '''
        if os.path.exists(fpath):
            try:
                model = joblib.load(os.path.join(fpath, 'RF_model.pkl'))
            except:
                raise FileExistsError("The dictionary {} doesn't exist model pkl file".format(fpath))
            return model
        else:
            raise FileExistsError("The dictionary {} doesn't exist model files".format(fpath))

    def save_model(self, fpath):
        '''
        Save model into the dictionary 'fpath'.
        The model will be saved into the 'fpath' in format:
            if keras: '_model_architecture.json' and '_model_weights.h5'
            if sklearn: '_model.pkl'

        :param fpath: the dictionary of saving-model
        :return:
        '''
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        joblib.dump(self.model, os.path.join(fpath, 'RF_model.pkl'))
        print("The RandomForest Model save in \n  {}".format(os.path.join(fpath, 'RF_model.pkl')))

    def evalution_with_data(self, X_test, Y_test):
        '''
        Evalution model with data 'X_test' and 'Y_test'

        :param X_test: the data using into model.predict
        :param Y_test: the true target data
        :return: Accuracy, Precision, Recall, F1, TPR, FPR, AUC
        '''
        Y_pred = self.model.predict(X_test)
        return self.calculate(Y_test, Y_pred)

    def predict(self, X_test):
        '''
        Predict data 'X_test' and return the classified output

        :param X_test: the data using into model.predict
        :return: np.array of classifed data(0 or 1)
        '''
        Y_pred = self.model.predict(X_test)
        return np.round(Y_pred).reshape(-1)

    def predict_proba(self, X_test):
        '''
        Predict data 'X_test' and return the probability output

        :param X_test: the data using into model.predict_proba
        :return: np.array of probability data([0, 1])
        '''
        Y_pred = self.model.predict_proba(X_test)[:,1]
        return Y_pred.reshape(-1)