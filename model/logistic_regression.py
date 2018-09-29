from .base import ModelType, AbstractModel

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
from keras.models import model_from_json
import os
import numpy as np

class LogisticRegression(AbstractModel):
    def __init__(self,
                 X_train = None,
                 Y_train = None,
                 X_test = None,
                 Y_test = None,
                 fpath = None):
        '''
        The LogisticRegression model class

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

        self.type = ModelType.LR

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
        input = self.X_train.shape[1]
        model = Sequential()
        reg = L1L2(l1=0.01, l2=0.01)
        model.add(Dense(1, input_dim=input, activation="sigmoid", kernel_regularizer=reg))
        model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
        return model

    def evalution(self, is_GA = False):
        '''
        Evalution model with data 'self.X_test' and 'self.Y_test'

        :param is_GA: True or False. Whether using by Genetic Algorithm
        :return:
            if is_GA is True: AUC
            if is_GA is False: Accuracy, Precision, Recall, F1, TPR, FPR, AUC
        '''
        if is_GA:
            self.model.fit(self.X_train, self.Y_train, epochs=5, batch_size=128, verbose=0)
        else:
            self.model.fit(self.X_train, self.Y_train, epochs=100, batch_size=20, verbose=1)

        Y_pred = self.model.predict(self.X_test)
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
                with open(os.path.join(fpath, 'LR_model_architecture.json'), 'r') as f:
                    model = model_from_json(f.read())
            except:
                raise FileExistsError("The dictionary {} doesn't exist model json file".format(fpath))
            try:
                model.load_weights(os.path.join(fpath, 'LR_model_weights.h5'))
            except:
                raise FileExistsError("The dictionary {} doesn't exist model h5 file".format(fpath))
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
        self.model.save_weights(os.path.join(fpath, 'LR_model_weights.h5'))
        with open(os.path.join(fpath, 'LR_model_architecture.json'), 'w') as f:
            f.write(self.model.to_json())
        print("The LogisticRegression Model save in \n  {} and \n  {}".format(
            os.path.join(fpath, 'LR_model_weights.h5'),
            os.path.join(fpath, 'LR_model_architecture.json')
        ))

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
        Y_pred = self.model.predict_proba(X_test)
        return Y_pred.reshape(-1)