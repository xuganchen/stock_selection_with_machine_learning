from model import *
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle as pkl
from portfolio import calcu_portfolio
from portfolio import plot_frequency, plot_model

np.random.seed(12345)


# ------------------------------ Training Model ------------------------------

modelpath = os.path.join("F:\\DeepLearning\\Model", datetime.now().strftime("%Y%m%d-%H%M%S"))



fpath = "F:\\DeepLearning\\Data"
fpath_insample = os.path.join(fpath, "insample")
fpath_outsample = os.path.join(fpath, "outsample")
X_train = np.load(os.path.join(fpath_insample, "X.npy"))
Y_train = np.load(os.path.join(fpath_insample, "Y.npy"))
X_test = np.load(os.path.join(fpath_outsample, "X.npy"))
Y_test = np.load(os.path.join(fpath_outsample, "Y.npy"))


# with open("F:\\DeepLearning\\Model\\result_GA.pkl", "rb") as file:
#     GA = pkl.load(file)
# GA_factor = GA['best_factors']
# X_train_GA = X_train[:, GA_factor == 1].shape
# Y_train_GA = Y_train[:, GA_factor == 1].shape
# X_test_GA = X_test[:, GA_factor == 1].shape
# Y_test_GA = Y_test[:, GA_factor == 1].shape

results = pd.DataFrame(index=['LG_before', 'RF_before', 'SVM_before', 'DNN_before',
                              'LG_after', 'RF_after', 'SVM_after', 'DNN_after'],
                       columns=['Accuracy', 'Precision', 'Recall', 'F1', 'TPR', 'FPR', 'AUC'])


# begore GA

lg = LogisticRegression(X_train, Y_train, X_test, Y_test)
print("\n\n", "Begin Model: {}\n\n".format(lg.type))
results.loc['LG_GA_before'] = lg.evalution()
print("\n\nThe results of LG model:")
print(results.loc['LG_GA_before'])
lg.save_model(os.path.join(modelpath, "GA_before"))


rf = RandomForest(X_train, Y_train, X_test, Y_test)
print("\n\n", "Begin Model: {}\n\n".format(rf.type))
results.loc['RF_GA_before'] = rf.evalution()
print("\n\nThe results of RF model:")
print(results.loc['RF_GA_before'])
rf.save_model(os.path.join(modelpath, "GA_before"))


svm = SupportVectorMachine(X_train, Y_train, X_test, Y_test)
print("\n\n", "Begin Model: {}\n\n".format(svm.type))
results.loc['SVM_GA_before'] = svm.evalution()
print("\n\nThe results of SVM model:")
print(results.loc['SVM_GA_before'])
svm.save_model(os.path.join(modelpath, "GA_before"))


dnn = DNN(X_train, Y_train, X_test, Y_test)
print("\n\n", "Begin Model: {}\n\n".format(dnn.type))
results.loc['DNN_GA_before'] = dnn.evalution()
print("\n\nThe results of DNN model:")
print(results.loc['DNN_GA_before'])
dnn.save_model(os.path.join(modelpath, "GA_after"))


# # after GA
# lg = LogisticRegression(X_train_GA, Y_train_GA, X_test_GA, Y_test_GA)
# print("\n\n", "Begin Model: {}\n\n".format(lg.type))
# results.loc['LG_GA_after'] = lg.evalution()
# print("\n\nThe results of LG model:")
# print(results.loc['LG_GA_after'])
# lg.save_model(os.path.join(modelpath, "GA_after"))
#
#
# rf = RandomForest(X_train_GA, Y_train_GA, X_test_GA, Y_test_GA)
# print("\n\n", "Begin Model: {}\n\n".format(rf.type))
# results.loc['RF_GA_after'] = rf.evalution()
# print("\n\nThe results of RF model:")
# print(results.loc['RF_GA_after'])
# rf.save_model(os.path.join(modelpath, "GA_after"))
#
#
# svm = SupportVectorMachine(X_train_GA, Y_train_GA, X_test_GA, Y_test_GA)
# print("\n\n", "Begin Model: {}\n\n".format(svm.type))
# results.loc['SVM_GA_after'] = svm.evalution()
# print("\n\nThe results of SVM model:")
# print(results.loc['SVM_GA_after'])
# svm.save_model(os.path.join(modelpath, "GA_after"))
#
#
# dnn = DNN(X_train_GA, Y_train_GA, X_test_GA, Y_test_GA)
# print("\n\n", "Begin Model: {}\n\n".format(dnn.type))
# results.loc['DNN_GA_after'] = dnn.evalution()
# print("\n\nThe results of DNN model:")
# print(results.loc['DNN_GA_after'])
# dnn.save_model(os.path.join(modelpath, "GA_after"))

with open(os.path.join(modelpath, "results.pkl"), 'wb+') as file:
    pkl.dump(results, file)

# with open(os.path.join(modelpath, "results.pkl"), "rb") as file:
#     results = pkl.load(file)



# ------------------------------ Calculate Portfolio ------------------------------
#
# modelpath = "F:\\DeepLearning\\Model\\20181002-005913"
# savedirpath = os.path.join(modelpath, "results")
# fpath = "F:\\DeepLearning\\data\\outsample_total"
#
# lg = LogisticRegression(fpath=modelpath)
# rf = RandomForest(fpath=modelpath)
# svm = SupportVectorMachine(fpath=modelpath)
# dnn = DNN(fpath=modelpath)
#
# modelsList = [lg, rf, svm, dnn]
# frequencyList = [3, 5, 7, 10, 15, 30]
#
# all_equitys = {}
#
# for model in modelsList:
#     equitys = {}
#     savepath = os.path.join(savedirpath, model.type.name)
#     if not os.path.exists(savepath):
#         os.makedirs(savepath)
#
#     for frequency in frequencyList:
#         X_portfolio = np.load(os.path.join(fpath, "factors_" + str(frequency) + "days.npy"))
#         Y_portfolio = np.load(os.path.join(fpath, "prices_" + str(frequency) + "days.npy"))
#         fname = model.type.name + "_" + str(frequency) + "days"
#         equity = calcu_portfolio(model, X_portfolio, Y_portfolio, savepath=savepath, fname=fname, isshow=False)
#         equitys[str(frequency) + "days"] = equity
#
#     with open(os.path.join(savepath, "equitys.pkl"), "wb+") as file:
#         pkl.dump(equitys, file)
#
#     all_equitys[model.type.name] = equitys
#     plot_frequency(equitys, model.type.name + "_frequency", savepath=savepath, isshow=False)
#
# frequencyListstr = []
# for i in frequencyList:
#     frequencyListstr.append(str(i) + "days")
# for freq in frequencyListstr:
#     equitys = {}
#     for name in all_equitys:
#         equitys[name] = all_equitys[name][freq]
#     plot_model(equitys,  "model_" + freq,  savepath=savedirpath, isshow=False)
#
# with open(os.path.join(savedirpath, "all_equitys.pkl"), "wb+") as file:
#     pkl.dump(all_equitys, file)