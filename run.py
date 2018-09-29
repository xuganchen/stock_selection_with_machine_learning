from model import *
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle as pkl

np.random.seed(12345)


modelpath = os.path.join("F:\\DeepLearning\\Model", datetime.now().strftime("%Y%m%d-%H%M%S"))

fpath = "F:\\DeepLearning\\Data"
fpath_insample = os.path.join(fpath, "insample")
fpath_outsample = os.path.join(fpath, "outsample")
X_train = np.load(os.path.join(fpath_insample, "X.npy"))
Y_train = np.load(os.path.join(fpath_insample, "Y.npy"))
X_test = np.load(os.path.join(fpath_outsample, "X.npy"))
Y_test = np.load(os.path.join(fpath_outsample, "Y.npy"))


# fpath = "F:\\DeepLearning\\Data\\sample"
# X = np.load(os.path.join(fpath, "X.npy"))
# Y = np.load(os.path.join(fpath, "Y.npy"))
# X_train = X[:800,:]
# Y_train = Y[:800]
# X_test = X[800:,:]
# Y_test = Y[800:]


results = pd.DataFrame(index=['LG', 'RF', 'SVM', 'DNN'],
                       columns=['Accuracy', 'Precision', 'Recall', 'F1', 'TPR', 'FPR', 'AUC'])

lg = LogisticRegression(X_train, Y_train, X_test, Y_test)
print("\n\n", "Begin Model: {}\n\n".format(lg.type))
results.loc['LG'] = lg.evalution()
print("\n\nThe results of LG model:")
print(results.loc['LG'])
lg.save_model(modelpath)


rf = RandomForest(X_train, Y_train, X_test, Y_test)
print("\n\n", "Begin Model: {}\n\n".format(rf.type))
results.loc['RF'] = rf.evalution()
print("\n\nThe results of RF model:")
print(results.loc['RF'])
rf.save_model(modelpath)


svm = SupportVectorMachine(X_train, Y_train, X_test, Y_test)
print("\n\n", "Begin Model: {}\n\n".format(svm.type))
results.loc['SVM'] = svm.evalution()
print("\n\nThe results of SVM model:")
print(results.loc['SVM'])
svm.save_model(modelpath)


dnn = DNN(X_train, Y_train, X_test, Y_test)
print("\n\n", "Begin Model: {}\n\n".format(dnn.type))
results.loc['DNN'] = dnn.evalution()
print("\n\nThe results of DNN model:")
print(results.loc['DNN'])
dnn.save_model(modelpath)

with open(os.path.join(modelpath, "results.pkl"), 'wb+') as file:
        pkl.dump(results, file)

# with open(os.path.join(modelpath, "results.pkl"), "rb") as file:
#     results = pkl.load(file)