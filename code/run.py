try:
    from .model import *
    from .portfolio import plot_multi, prediction
    from .portfolio import backtesting, plot_backtest, get_results
    from .portfolio import get_avg_weight
    from .blacklitterman import BlackLitterman
except:
    from model import *
    from portfolio import plot_multi, prediction
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


# # ------------------------------ Using BlackLitterman Model ------------------------------
#
# # pred_data_kind="bool"
# pred_data_kind="proba"
#
# modelpath = os.path.join("F:\\DeepLearning\\Model\\20181004-214808")
# datapath = "F:\\DeepLearning\\data\\outsample_total"
# if pred_data_kind == "bool":
#     resultspath = os.path.join("F:\\DeepLearning\\Model\\20181004-214808\\results_from_bool")
# elif pred_data_kind == "proba":
#     resultspath = os.path.join("F:\\DeepLearning\\Model\\20181004-214808\\results_from_probas")
# avgresultspath = os.path.join("F:\\DeepLearning\\Model\\20181004-214808\\results_avg")
#
# if not os.path.exists(resultspath):
#     os.makedirs(resultspath)
# if not os.path.exists(avgresultspath):
#     os.makedirs(avgresultspath)
#
# GAtypeList = ['beforeGA', 'afterGA']
# # frequencyList = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30]
# frequencyList = [10, 12, 15, 18, 20, 25, 30]
# modelList = {
#     'LR': LogisticRegression,
#     'RF': RandomForest,
#     'SVM': SupportVectorMachine,
#     'DNN': DNN,
#     'NB': NaiveBeyes,
#     'GBM': GradientBoostingMachine,
#     'BAG': Bagging,
#     'ET': ExtraTrees,
#     'ADA': AdaBoost,
#     'EXS': EnsembleXgbStack
# }
# init_equity = 100000
#
# ### For BlackLitterman Model
#
# return_data = pd.read_hdf(os.path.join(datapath, "return_data.h5"))
# mean_cum_returns = pd.read_hdf(os.path.join(datapath, "mean_cum_returns.h5"))
# market_data = pd.read_hdf(os.path.join(datapath, "market_data.h5"))
# risk_free = pd.read_hdf(os.path.join(datapath, "risk_free.h5"))
#
# for GAtype in GAtypeList:
#     saveGApath = os.path.join(modelpath, GAtype)
#     for modelname in modelList.keys():
#         savemodelpath = os.path.join(saveGApath, modelname)
#         saveresultspath = os.path.join(resultspath, modelname)
#         if not os.path.exists(saveresultspath):
#             os.makedirs(saveresultspath)
#
#         for frequency in frequencyList:
#             Y_portfolio = np.load(os.path.join(datapath, "prices_" + str(frequency) + "days.npy"))
#             benchmark_returns = np.load(os.path.join(datapath, "benchmark_returns_" + str(frequency) + "days.npy"))
#             days = np.load(os.path.join(datapath, "todays_" + str(frequency) + "days.npy"))
#             if pred_data_kind == "bool":
#                 fname = modelname + "_" + str(frequency) + "days_portfolios.npy"
#             elif pred_data_kind == "proba":
#                 fname = modelname + "_" + str(frequency) + "days_portfolio_probas.npy"
#             pred_data = np.load(os.path.join(savemodelpath, fname))
#
#             bl = BlackLitterman(return_data, mean_cum_returns, market_data,
#                                 risk_free, pred_data, days, frequency,
#                                 pred_data_kind=pred_data_kind)
#             weights = bl.get_weights()
#             pngname = "BL_" + GAtype + "_" + modelname + "_" + str(frequency) + "days"
#             np.save(os.path.join(saveresultspath, pngname + ".npy"), weights)
#             equitys = backtesting(weights, Y_portfolio, init_equity=init_equity)
#             bench_equity = np.concatenate(([init_equity],benchmark_returns.cumprod() * init_equity))
#             days = pd.Series(np.concatenate((days, [20180629]))).apply(lambda x: pd.Timestamp.strptime(str(x), "%Y%m%d"))
#             equitys = equity = pd.Series(equitys, index=days)
#             bench_equity = pd.Series(bench_equity, index=days)
#
#             plot_backtest(equitys, bench_equity, fname = pngname, savepath = saveresultspath)
#             results = get_results(equitys, bench_equity, frequency)
#
#             with open(os.path.join(saveresultspath, pngname + ".pkl"), "wb+") as file:
#                 pkl.dump(results, file)
#             print("Done: ", pngname)



# ### For Average Weight
#
# for GAtype in GAtypeList:
#     saveGApath = os.path.join(modelpath, GAtype)
#     for modelname in modelList.keys():
#         savemodelpath = os.path.join(saveGApath, modelname)
#         saveresultspath = os.path.join(avgresultspath, modelname)
#         if not os.path.exists(saveresultspath):
#             os.makedirs(saveresultspath)
#
#         for frequency in frequencyList:
#             fname = modelname + "_" + str(frequency) + "days_portfolios.npy"
#             benchmark_returns = np.load(os.path.join(datapath, "benchmark_returns_" + str(frequency) + "days.npy"))
#             days = np.load(os.path.join(datapath, "todays_" + str(frequency) + "days.npy"))
#             pred_data = np.load(os.path.join(savemodelpath, fname))
#             Y_portfolio = np.load(os.path.join(datapath, "prices_" + str(frequency) + "days.npy"))
#
#             weights = get_avg_weight(pred_data)
#             pngname = "AVG_" + GAtype + "_" + modelname + "_" + str(frequency) + "days"
#             np.save(os.path.join(saveresultspath, pngname + ".npy"), weights)
#
#             equitys = backtesting(weights, Y_portfolio)
#             bench_equity = np.concatenate(([init_equity],benchmark_returns.cumprod() * init_equity))
#             days = pd.Series(np.concatenate((days, [20180629]))).apply(lambda x: pd.Timestamp.strptime(str(x), "%Y%m%d"))
#             equitys = equity = pd.Series(equitys, index=days)
#             bench_equity = pd.Series(bench_equity, index=days)
#
#             plot_backtest(equitys, bench_equity, fname = pngname, savepath = saveresultspath)
#             results = get_results(equitys, bench_equity, frequency)
#
#             with open(os.path.join(saveresultspath, pngname + ".pkl"), "wb+") as file:
#                 pkl.dump(results, file)
#             print("Done: ", pngname)



# # ------------------------------ Plot Result ------------------------------
#
# probasresultspath = os.path.join("F:\\DeepLearning\\Model\\20181004-214808\\results_from_probas")
# boolresultspath = os.path.join("F:\\DeepLearning\\Model\\20181004-214808\\results_from_bool")
# avgresultspath = os.path.join("F:\\DeepLearning\\Model\\20181004-214808\\results_avg")
# datapath = "F:\\DeepLearning\\data\\outsample_total"
# savepath = "F:\\DeepLearning\\Model\\20181004-214808\\results_from_probas\\multi"
#
#
# GAtypeList = ['beforeGA', 'afterGA']
# # frequencyList = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30]
# frequencyList = [10, 12, 15, 18, 20, 25, 30]
# modelList = ['LR', 'RF', 'SVM', 'DNN', 'NB', 'GBM', 'BAG', 'ET', 'ADA', 'EXS']
# init_equity = 100000
#
#
# def get_bench_equity(frequency, datapath, init_equity):
#     days = np.load(os.path.join(datapath, "todays_" + str(frequency) + "days.npy"))
#     days = pd.Series(np.concatenate((days, [20180629]))).apply(lambda x: pd.Timestamp.strptime(str(x), "%Y%m%d"))
#     benchmark_returns = np.load(os.path.join(datapath, "benchmark_returns_" + str(frequency) + "days.npy"))
#
#     bench_equity = np.concatenate(([init_equity], benchmark_returns.cumprod() * init_equity))
#     bench_equity = pd.Series(bench_equity, index=days)
#     return bench_equity
#
#
# ## "By_frequency"
# savepathBy_frequency = os.path.join(savepath, "By_frequency")
# for GAtype in GAtypeList:
#     for modelname in modelList:
#         probasequity = {}
#         saveresultspath = os.path.join(probasresultspath, modelname)
#         for frequency in frequencyList:
#             pngname = "BL_" + GAtype + "_" + modelname + "_" + str(frequency) + "days"
#             try:
#                 with open(os.path.join(saveresultspath, pngname + ".pkl"), "rb") as file:
#                     result = pkl.load(file)
#                 probasequity[pngname] = result['equity']
#             except:
#                 pass
#         fname = "BL_" + GAtype + "_" + modelname
#         plot_multi(probasequity, fname, savepathBy_frequency)
#
# ## "By_model"
# savepathBy_model = os.path.join(savepath, "By_model")
# for GAtype in GAtypeList:
#     for frequency in frequencyList:
#         probasequity = {}
#         bench_equity = get_bench_equity(20, datapath, init_equity)
#         probasequity['HS300_' + str(frequency) + "days"] = bench_equity
#         for modelname in modelList:
#             saveresultspath = os.path.join(probasresultspath, modelname)
#             pngname = "BL_" + GAtype + "_" + modelname + "_" + str(frequency) + "days"
#             try:
#                 with open(os.path.join(saveresultspath, pngname + ".pkl"), "rb") as file:
#                     result = pkl.load(file)
#                 probasequity[pngname] = result['equity']
#             except:
#                 pass
#         fname = "BL_" + GAtype + "_" + str(frequency) + "days"
#         plot_multi(probasequity, fname, savepathBy_model)
#
# ## "By_GAtype"
# savepathBy_GAtype = os.path.join(savepath, "By_GAtype")
# for frequency in frequencyList:
#     for modelname in modelList:
#         probasequity = {}
#         bench_equity = get_bench_equity(20, datapath, init_equity)
#         probasequity['HS300_' + str(frequency) + "days"] = bench_equity
#         for GAtype in GAtypeList:
#             saveresultspath = os.path.join(probasresultspath, modelname)
#             pngname = "BL_" + GAtype + "_" + modelname + "_" + str(frequency) + "days"
#             try:
#                 with open(os.path.join(saveresultspath, pngname + ".pkl"), "rb") as file:
#                     result = pkl.load(file)
#                 probasequity[pngname] = result['equity']
#             except:
#                 pass
#         fname = "BL_" + modelname + "_" + str(frequency) + "days"
#         plot_multi(probasequity, fname, savepathBy_GAtype)
#
# ## "Total Return for Model"
# fig = plt.figure(figsize=(16, 9))
# gs = gridspec.GridSpec(1, 2, wspace=0.25, hspace=0.5)
# ax1 = plt.subplot(gs[:, 0])
# beforeGA = probasresults.loc[probasresults['is_GA'] == "beforeGA"][['tot_return', 'model', 'frequency']]
# beforeGA_tot_return = beforeGA.set_index(['model', 'frequency']).unstack()
# beforeGA_tot_return.columns = beforeGA_tot_return.columns.levels[1]
# sns.heatmap(
#     beforeGA_tot_return,
#     fmt="0.2%",
#     annot_kws={"size": 8},
#     alpha=1.0,
#     center=0.0,
#     cbar=False,
#     cmap=cm.RdYlGn,
#     annot=True,
#     ax=ax1)
# ax1.set_title('Total Return for Model before GA')
#
# ax2 = plt.subplot(gs[:, 1])
# afterGA = probasresults.loc[probasresults['is_GA'] == "afterGA"][['tot_return', 'model', 'frequency']]
# afterGA_tot_return = afterGA.set_index(['model', 'frequency']).unstack()
# afterGA_tot_return.columns = afterGA_tot_return.columns.levels[1]
# sns.heatmap(
#     afterGA_tot_return,
#     fmt="0.2%",
#     annot_kws={"size": 8},
#     alpha=1.0,
#     center=0.0,
#     cbar=False,
#     cmap=cm.RdYlGn,
#     annot=True,
#     ax=ax2)
# ax2.set_title('Total Return for Model after GA')
#
# fname = "Total Return for Model"
# fig.savefig(os.path.join(savepath, fname))
#
#
# ## "Annual Return for Model"
# fig = plt.figure(figsize=(16, 9))
# gs = gridspec.GridSpec(1, 2, wspace=0.25, hspace=0.5)
#
# ax1 = plt.subplot(gs[:, 0])
# beforeGA = probasresults.loc[probasresults['is_GA'] == "beforeGA"][['annual_return', 'model', 'frequency']]
# beforeGA_annual_return = beforeGA.set_index(['model', 'frequency']).unstack()
# beforeGA_annual_return.columns = beforeGA_annual_return.columns.levels[1]
# sns.heatmap(
#     beforeGA_annual_return,
#     fmt="0.2%",
#     annot_kws={"size": 8},
#     alpha=1.0,
#     center=0.0,
#     cbar=False,
#     cmap=cm.RdYlGn,
#     annot=True,
#     ax=ax1)
# ax1.set_title('Annual Return for Model before GA')
#
# ax2 = plt.subplot(gs[:, 1])
# afterGA = probasresults.loc[probasresults['is_GA'] == "afterGA"][['annual_return', 'model', 'frequency']]
# afterGA_annual_return = afterGA.set_index(['model', 'frequency']).unstack()
# afterGA_annual_return.columns = afterGA_annual_return.columns.levels[1]
# sns.heatmap(
#     afterGA_annual_return,
#     fmt="0.2%",
#     annot_kws={"size": 8},
#     alpha=1.0,
#     center=0.0,
#     cbar=False,
#     cmap=cm.RdYlGn,
#     annot=True,
#     ax=ax2)
# ax2.set_title('Annual Return for Model after GA')
#
# fname = "Annual Return for Model"
# fig.savefig(os.path.join(savepath, fname))
#
#
# ## "Sharpe for Model"
# fig = plt.figure(figsize=(16, 9))
# gs = gridspec.GridSpec(1, 2, wspace=0.25, hspace=0.5)
#
# ax1 = plt.subplot(gs[:, 0])
# beforeGA = probasresults.loc[probasresults['is_GA'] == "beforeGA"][['sharpe', 'model', 'frequency']]
# beforeGA_sharpe = beforeGA.set_index(['model', 'frequency']).unstack()
# beforeGA_sharpe.columns = beforeGA_sharpe.columns.levels[1]
# sns.heatmap(
#     beforeGA_sharpe,
#     fmt="0.3",
#     annot_kws={"size": 8},
#     alpha=1.0,
#     center=0.0,
#     cbar=False,
#     cmap=cm.RdYlGn,
#     annot=True,
#     ax=ax1)
# ax1.set_title('Sharpe for Model before GA')
#
# ax2 = plt.subplot(gs[:, 1])
# afterGA = probasresults.loc[probasresults['is_GA'] == "afterGA"][['sharpe', 'model', 'frequency']]
# afterGA_sharpe = afterGA.set_index(['model', 'frequency']).unstack()
# afterGA_sharpe.columns = afterGA_sharpe.columns.levels[1]
# sns.heatmap(
#     afterGA_sharpe,
#     fmt="0.3",
#     annot_kws={"size": 8},
#     alpha=1.0,
#     center=0.0,
#     cbar=False,
#     cmap=cm.RdYlGn,
#     annot=True,
#     ax=ax2)
# ax2.set_title('Sharpe for Model after GA')
#
# fname = "Sharpe for Model"
# fig.savefig(os.path.join(savepath, fname))
#
#
# ## "Max Drawdown for Model"
# fig = plt.figure(figsize=(16, 9))
# gs = gridspec.GridSpec(1, 2, wspace=0.25, hspace=0.5)
#
# ax1 = plt.subplot(gs[:, 0])
# beforeGA = probasresults.loc[probasresults['is_GA'] == "beforeGA"][['max_drawdown', 'model', 'frequency']]
# beforeGA_max_drawdown = beforeGA.set_index(['model', 'frequency']).unstack()
# beforeGA_max_drawdown.columns = beforeGA_max_drawdown.columns.levels[1]
# sns.heatmap(
#     beforeGA_max_drawdown,
#     fmt="0.2%",
#     annot_kws={"size": 8},
#     alpha=1.0,
#     center=0.0,
#     cbar=False,
#     cmap=cm.RdYlGn,
#     annot=True,
#     ax=ax1)
# ax1.set_title('Max Drawdown for Model before GA')
#
# ax2 = plt.subplot(gs[:, 1])
# afterGA = probasresults.loc[probasresults['is_GA'] == "afterGA"][['max_drawdown', 'model', 'frequency']]
# afterGA_max_drawdown = afterGA.set_index(['model', 'frequency']).unstack()
# afterGA_max_drawdown.columns = afterGA_max_drawdown.columns.levels[1]
# sns.heatmap(
#     afterGA_max_drawdown,
#     fmt="0.2%",
#     annot_kws={"size": 8},
#     alpha=1.0,
#     center=0.0,
#     cbar=False,
#     cmap=cm.RdYlGn,
#     annot=True,
#     ax=ax2)
# ax2.set_title('Max Drawdown for Model after GA')
#
# fname = "Max Drawdown for Model"
# fig.savefig(os.path.join(savepath, fname))