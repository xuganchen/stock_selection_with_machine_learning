from model import *
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle as pkl
from portfolio import plot_frequency, plot_model, prediction

np.random.seed(12345)


# ------------------------------ Training Model ------------------------------

modelpath = os.path.join("F:\\DeepLearning\\Model", datetime.now().strftime("%Y%m%d-%H%M%S"))
preddatapath = "F:\\DeepLearning\\data\\outsample_total"


fpath = "F:\\DeepLearning\\Data"
fpath_insample = os.path.join(fpath, "insample")
fpath_outsample = os.path.join(fpath, "outsample")
X_train = np.load(os.path.join(fpath_insample, "X.npy"))
Y_train = np.load(os.path.join(fpath_insample, "Y.npy"))
X_test = np.load(os.path.join(fpath_outsample, "X.npy"))
Y_test = np.load(os.path.join(fpath_outsample, "Y.npy"))


with open("F:\\DeepLearning\\Model\\result_GA.pkl", "rb") as file:
    GA = pkl.load(file)
GA_factor = GA['best_factors']
X_train_GA = X_train[:, GA_factor == 1]
Y_train_GA = Y_train
X_test_GA = X_test[:, GA_factor == 1]
Y_test_GA = Y_test

results = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1', 'TPR', 'FPR', 'AUC'])

modelList = {
    'lr': LogisticRegression,
    'rf': RandomForest,
    'svm': SupportVectorMachine,
    'dnn': DNN,
    'nb': NaiveBeyes,
    'gbm': GradientBoostingMachine,
    'bag': Bagging,
    'et': ExtraTrees,
    'ada': AdaBoost,
    'exs': EnsembleXgbStack
}

frequencyList = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30]


# before GA
GAtype = "beforeGA"
savemodelpath = os.path.join(modelpath, GAtype)

for modelname in modelList.keys():
    model = modelList[modelname](X_train, Y_train, X_test, Y_test)
    print("\n\n", "Begin Model: {}".format(model.type))
    results.loc[model.type.name + "_" + GAtype] = model.evalution()
    print("The results of " + model.type.name + " model:")
    print(results.loc[model.type.name + "_" + GAtype])

    savepath = os.path.join(savemodelpath, model.type.name)
    model.save_model(savepath)

    for frequency in frequencyList:
        X_portfolio = np.load(os.path.join(preddatapath, "factors_" + str(frequency) + "days.npy"))
        fname = model.type.name + "_" + str(frequency) + "days"
        portfolios, portfolio_probas = prediction(model, X_portfolio, savepath, fname)



# after GA
GAtype = "afterGA"
savemodelpath = os.path.join(modelpath, GAtype)

for modelname in modelList.keys():
    model = modelList[modelname](X_train_GA, Y_train_GA, X_test_GA, Y_test_GA)
    print("\n\n", "Begin Model: {}".format(model.type))
    results.loc[model.type.name + "_" + GAtype] = model.evalution()
    print("The results of " + model.type.name + " model:")
    print(results.loc[model.type.name + "_" + GAtype])

    savepath = os.path.join(savemodelpath, model.type.name)
    model.save_model(savepath)

    for frequency in frequencyList:
        X_portfolio = np.load(os.path.join(preddatapath, "factors_" + str(frequency) + "days.npy"))
        X_portfolio = np.array([port[:, GA_factor == 1] for port in X_portfolio])
        fname = model.type.name + "_" + str(frequency) + "days"
        portfolios, portfolio_probas = prediction(model, X_portfolio, savepath, fname)


with open(os.path.join(modelpath, "results.pkl"), 'wb+') as file:
    pkl.dump(results, file)

# with open(os.path.join(modelpath, "results.pkl"), "rb") as file:
#     results = pkl.load(file)


