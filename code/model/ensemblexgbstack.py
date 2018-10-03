from .base import ModelType, AbstractModel
from . import AdaBoost, Bagging, ExtraTrees, GradientBoostingMachine, RandomForest

import xgboost as xgb
from sklearn.externals import joblib
import os
import numpy as np

class EnsembleXgbStack(AbstractModel):
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

        self.type = ModelType.EXS

        if fpath is None:
            self.X_train = X_train
            self.Y_train = Y_train
            self.X_test = X_test
            self.Y_test = Y_test
            self.leng = self.X_train.shape[0] // 3
            self.model = self._generate_model()
        else:
            self.model = self._load_model(fpath)

    def _generate_model(self):
        '''
        Generate model with empty class
        '''
        X_train1 = self.X_train[: -self.leng, :]
        Y_train1 = self.Y_train[: -self.leng]
        X_train2 = self.X_train[-self.leng:, :]
        Y_train2 = self.Y_train[-self.leng:]

        self.ada = AdaBoost(X_train1, Y_train1, X_train2, Y_train2)
        self.bag = Bagging(X_train1, Y_train1, X_train2, Y_train2)
        self.et = ExtraTrees(X_train1, Y_train1, X_train2, Y_train2)
        self.gb = GradientBoostingMachine(X_train1, Y_train1, X_train2, Y_train2)
        self.rf = RandomForest(X_train1, Y_train1, X_train2, Y_train2)

        self.ada.model.fit(X_train1, Y_train1)
        self.bag.model.fit(X_train1, Y_train1)
        self.et.model.fit(X_train1, Y_train1)
        self.gb.model.fit(X_train1, Y_train1)
        self.rf.model.fit(X_train1, Y_train1)

        model = xgb.XGBRegressor(max_depth=7, objective="reg:logistic", learning_rate=0.5, n_estimators=500)
        return model

    def _generate_X(self, X_train2):
        Y1 = self.ada.predict_proba(X_train2).reshape(-1, 1)
        Y2 = self.bag.predict_proba(X_train2).reshape(-1, 1)
        Y3 = self.et.predict_proba(X_train2).reshape(-1, 1)
        Y4 = self.gb.predict_proba(X_train2).reshape(-1, 1)
        Y5 = self.rf.predict_proba(X_train2).reshape(-1, 1)
        X_frompred = np.concatenate((Y1, Y2, Y3, Y4, Y5), axis=1)
        return X_frompred


    def evalution(self, is_GA = False):
        '''
        Evalution model with data 'self.X_test' and 'self.Y_test'

        :param is_GA: True or False. Whether using by Genetic Algorithm
        :return:
            if is_GA is True: AUC
            if is_GA is False: Accuracy, Precision, Recall, F1, TPR, FPR, AUC
        '''
        X_train2 = self.X_train[-self.leng:, :]
        Y_train2 = self.Y_train[-self.leng:]
        X_frompred = self._generate_X(X_train2)
        self.model.fit(X_frompred, Y_train2)

        X_test_frompred = self._generate_X(self.X_test)
        Y_pred = self.model.predict(X_test_frompred)
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
                self.ada = AdaBoost(fpath=fpath)
                self.bag = Bagging(fpath=fpath)
                self.et = ExtraTrees(fpath=fpath)
                self.gb = GradientBoostingMachine(fpath=fpath)
                self.rf = RandomForest(fpath=fpath)
                model = joblib.load(os.path.join(fpath, 'EXS_model.pkl'))
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
        self.ada.save_model(fpath)
        self.bag.save_model(fpath)
        self.et.save_model(fpath)
        self.gb.save_model(fpath)
        self.rf.save_model(fpath)
        joblib.dump(self.model, os.path.join(fpath, 'EXS_model.pkl'))
        print("The EnsembleXgbStack Model save in \n  {}".format(os.path.join(fpath, 'EXS_model.pkl')))

    def evalution_with_data(self, X_test, Y_test):
        '''
        Evalution model with data 'X_test' and 'Y_test'

        :param X_test: the data using into model.predict
        :param Y_test: the true target data
        :return: Accuracy, Precision, Recall, F1, TPR, FPR, AUC
        '''

        X_test_frompred = self._generate_X(X_test)
        Y_pred = self.model.predict(X_test_frompred)
        return self.calculate(Y_test, Y_pred)

    def predict(self, X_test):
        '''
        Predict data 'X_test' and return the classified output

        :param X_test: the data using into model.predict
        :return: np.array of classifed data(0 or 1)
        '''

        X_test_frompred = self._generate_X(X_test)
        Y_pred = self.model.predict(X_test_frompred)
        return np.round(Y_pred).reshape(-1)

    def predict_proba(self, X_test):
        '''
        Predict data 'X_test' and return the probability output

        :param X_test: the data using into model.predict_proba
        :return: np.array of probability data([0, 1])
        '''

        X_test_frompred = self._generate_X(X_test)
        Y_pred = self.model.predict(X_test_frompred)
        return Y_pred.reshape(-1)