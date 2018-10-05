try:
    from .model import *
    from .portfolio import plot_frequency, plot_model, prediction
    from .portfolio import backtesting, plot_backtest, get_results
    from .portfolio import get_avg_weight
    from .blacklitterman import BlackLitterman
except:
    from model import *
    from portfolio import plot_frequency, plot_model, prediction
    from portfolio import backtesting, plot_backtest, get_results
    from portfolio import get_avg_weight
    from blacklitterman import BlackLitterman

import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle as pkl

np.random.seed(12345)


# # ------------------------------ Training Model ------------------------------
#
# modelpath = os.path.join("F:\\DeepLearning\\Model", datetime.now().strftime("%Y%m%d-%H%M%S"))
# preddatapath = "F:\\DeepLearning\\data\\outsample_total"
#
#
# fpath = "F:\\DeepLearning\\Data"
# fpath_insample = os.path.join(fpath, "insample")
# fpath_outsample = os.path.join(fpath, "outsample")
# X_train = np.load(os.path.join(fpath_insample, "X.npy"))
# Y_train = np.load(os.path.join(fpath_insample, "Y.npy"))
# X_test = np.load(os.path.join(fpath_outsample, "X.npy"))
# Y_test = np.load(os.path.join(fpath_outsample, "Y.npy"))
#
#
# with open("F:\\DeepLearning\\Model\\result_GA.pkl", "rb") as file:
#     GA = pkl.load(file)
# GA_factor = GA['best_factors']
# X_train_GA = X_train[:, GA_factor == 1]
# Y_train_GA = Y_train
# X_test_GA = X_test[:, GA_factor == 1]
# Y_test_GA = Y_test
#
# results = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1', 'TPR', 'FPR', 'AUC'])
#
# modelList = {
#     'lr': LogisticRegression,
#     'rf': RandomForest,
#     'svm': SupportVectorMachine,
#     'dnn': DNN,
#     'nb': NaiveBeyes,
#     'gbm': GradientBoostingMachine,
#     'bag': Bagging,
#     'et': ExtraTrees,
#     'ada': AdaBoost,
#     'exs': EnsembleXgbStack
# }
#
# frequencyList = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30]

#
# # before GA
# GAtype = "beforeGA"
# savemodelpath = os.path.join(modelpath, GAtype)
#
# for modelname in modelList.keys():
#     model = modelList[modelname](X_train, Y_train, X_test, Y_test)
#     print("\n\n", "Begin Model: {}".format(model.type))
#     results.loc[model.type.name + "_" + GAtype] = model.evalution()
#     print("The results of " + model.type.name + " model:")
#     print(results.loc[model.type.name + "_" + GAtype])
#
#     savepath = os.path.join(savemodelpath, model.type.name)
#     model.save_model(savepath)
#
#     for frequency in frequencyList:
#         X_portfolio = np.load(os.path.join(preddatapath, "factors_" + str(frequency) + "days.npy"))
#         fname = model.type.name + "_" + str(frequency) + "days"
#         portfolios, portfolio_probas = prediction(model, X_portfolio, savepath, fname)



# # after GA
# GAtype = "afterGA"
# savemodelpath = os.path.join(modelpath, GAtype)
#
# for modelname in modelList.keys():
#     model = modelList[modelname](X_train_GA, Y_train_GA, X_test_GA, Y_test_GA)
#     print("\n\n", "Begin Model: {}".format(model.type))
#     results.loc[model.type.name + "_" + GAtype] = model.evalution()
#     print("The results of " + model.type.name + " model:")
#     print(results.loc[model.type.name + "_" + GAtype])
#
#     savepath = os.path.join(savemodelpath, model.type.name)
#     model.save_model(savepath)
#
#     for frequency in frequencyList:
#         X_portfolio = np.load(os.path.join(preddatapath, "factors_" + str(frequency) + "days.npy"))
#         X_portfolio = np.array([port[:, GA_factor == 1] for port in X_portfolio])
#         fname = model.type.name + "_" + str(frequency) + "days"
#         portfolios, portfolio_probas = prediction(model, X_portfolio, savepath, fname)
#
#
# with open(os.path.join(modelpath, "results_afterGA.pkl"), 'wb+') as file:
#     pkl.dump(results, file)
#
# # with open(os.path.join(modelpath, "results.pkl"), "rb") as file:
# #     results = pkl.load(file)


# ------------------------------ Using BlackLitterman Model ------------------------------

modelpath = os.path.join("F:\\DeepLearning\\Model\\20181004-214808")
datapath = "F:\\DeepLearning\\data\\outsample_total"
resultspath = os.path.join("F:\\DeepLearning\\Model\\20181004-214808\\results")
avgresultspath = os.path.join("F:\\DeepLearning\\Model\\20181004-214808\\results_avg")

if not os.path.exists(resultspath):
    os.makedirs(resultspath)
if not os.path.exists(avgresultspath):
    os.makedirs(avgresultspath)

GAtypeList = ['beforeGA', 'afterGA']
frequencyList = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30]
modelList = {
    'LR': LogisticRegression,
    'RF': RandomForest,
    'SVM': SupportVectorMachine,
    'DNN': DNN,
    'NB': NaiveBeyes,
    'GBM': GradientBoostingMachine,
    'BAG': Bagging,
    'ET': ExtraTrees,
    'ADA': AdaBoost,
    'EXS': EnsembleXgbStack
}
init_equity = 100000

### For BlackLitterman Model

return_data = pd.read_hdf(os.path.join(datapath, "return_data.h5"))
mean_cum_returns = pd.read_hdf(os.path.join(datapath, "mean_cum_returns.h5"))
market_data = pd.read_hdf(os.path.join(datapath, "market_data.h5"))
risk_free = pd.read_hdf(os.path.join(datapath, "risk_free.h5"))

for GAtype in GAtypeList:
    saveGApath = os.path.join(modelpath, GAtype)
    for modelname in modelList.keys():
        savemodelpath = os.path.join(saveGApath, modelname)
        saveresultspath = os.path.join(resultspath, modelname)

        for frequency in frequencyList:
            Y_portfolio = np.load(os.path.join(datapath, "prices_" + str(frequency) + "days.npy"))
            benchmark_returns = np.load(os.path.join(datapath, "benchmark_returns_" + str(frequency) + "days.npy"))
            days = np.load(os.path.join(datapath, "todays_" + str(frequency) + "days.npy"))
            # fname = modelname + "_" + str(frequency) + "days_portfolio_probas.npy"
            fname = modelname + "_" + str(frequency) + "days_portfolios.npy"
            pred_data = np.load(os.path.join(savemodelpath, fname))

            bl = BlackLitterman(return_data, mean_cum_returns, market_data,
                                risk_free, pred_data, days, frequency,
                                pred_data_kind="bool")
            weights = bl.get_weights()
            equitys = backtesting(weights, Y_portfolio, init_equity=init_equity)
            bench_equity = np.concatenate(([init_equity],benchmark_returns.cumprod() * init_equity))
            pngname = "BL_" + GAtype + "_" + modelname + "_" + str(frequency) + "days"
            plot_backtest(equitys, bench_equity, fname = pngname, savepath = saveresultspath)

            days = pd.Series(np.concatenate((days, [20180629]))).apply(lambda x: pd.Timestamp.strptime(str(x), "%Y%m%d"))
            equitys = equity = pd.Series(equitys, index=days)
            bench_equity = pd.Series(bench_equity, index=days)
            results = get_results(equitys, bench_equity, frequency)

            with open(os.path.join(saveresultspath, pngname + ".pkl"), "wb+") as file:
                pkl.dump(results, file)
            print("Done: ", pngname)



# ### For Average Weight
#
# for GAtype in GAtypeList:
#     saveGApath = os.path.join(modelpath, GAtype)
#     for modelname in modelList.keys():
#         savemodelpath = os.path.join(saveGApath, modelname)
#         saveresultspath = os.path.join(avgresultspath, modelname)
#
#         for frequency in frequencyList:
#             fname = modelname + "_" + str(frequency) + "days_portfolios.npy"
#             benchmark_returns = np.load(os.path.join(datapath, "benchmark_returns_" + str(frequency) + "days.npy"))
#             days = np.load(os.path.join(datapath, "todays_" + str(frequency) + "days.npy"))
#             pred_data = np.load(os.path.join(savemodelpath, fname))
#             Y_portfolio = np.load(os.path.join(datapath, "prices_" + str(frequency) + "days.npy"))
#
#             weights = get_avg_weight(pred_data)
#             equitys = backtesting(weights, Y_portfolio)
#             bench_equity = np.concatenate(([init_equity],benchmark_returns.cumprod() * init_equity))
#             pngname = "AVG_" + GAtype + "_" + modelname + "_" + str(frequency) + "days"
#             plot_backtest(equitys, bench_equity, fname = pngname, savepath = saveresultspath)
#
#
#             days = pd.Series(np.concatenate((days, [20180629]))).apply(lambda x: pd.Timestamp.strptime(str(x), "%Y%m%d"))
#             equitys = equity = pd.Series(equitys, index=days)
#             bench_equity = pd.Series(bench_equity, index=days)
#             results = get_results(equitys, bench_equity, frequency)
#
#             with open(os.path.join(saveresultspath, pngname + ".pkl"), "wb+") as file:
#                 pkl.dump(results, file)
#             print("Done: ", pngname)


