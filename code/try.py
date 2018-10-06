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

pred_data_kind="bool"
# pred_data_kind="proba"

modelpath = os.path.join("F:\\DeepLearning\\Model\\20181004-214808")
datapath = "F:\\DeepLearning\\data\\outsample_total"
if pred_data_kind == "bool":
    resultspath = os.path.join("F:\\DeepLearning\\Model\\20181004-214808\\results_from_bool")
elif pred_data_kind == "proba":
    resultspath = os.path.join("F:\\DeepLearning\\Model\\20181004-214808\\results_from_probas")
avgresultspath = os.path.join("F:\\DeepLearning\\Model\\20181004-214808\\results_avg")

# GAtypeList = ['beforeGA', 'afterGA']
GAtypeList = ['beforeGA']
# frequencyList = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30]
# frequencyList = [10, 15, 20, 25, 30]
frequencyList = [30]
modelList = {
#     'LR': LogisticRegression,
#     'RF': RandomForest,
#     'SVM': SupportVectorMachine,
    'DNN': DNN,
#     'NB': NaiveBeyes,
#     'GBM': GradientBoostingMachine,
#     'BAG': Bagging,
#     'ET': ExtraTrees,
#     'ADA': AdaBoost,
#     'EXS': EnsembleXgbStack
}
init_equity = 100000

return_data = pd.read_hdf(os.path.join(datapath, "return_data.h5"))
mean_cum_returns = pd.read_hdf(os.path.join(datapath, "mean_cum_returns.h5"))
market_data = pd.read_hdf(os.path.join(datapath, "market_data.h5"))
risk_free = pd.read_hdf(os.path.join(datapath, "risk_free.h5"))

GAtype = 'beforeGA'
frequency = 30
modelname = 'DNN'

a = 0.036
for i in np.arange(0.01, 0.1, 0.002):
    saveGApath = os.path.join(modelpath, GAtype)
    savemodelpath = os.path.join(saveGApath, modelname)
    saveresultspath = os.path.join(resultspath, modelname)

    Y_portfolio = np.load(os.path.join(datapath, "prices_" + str(frequency) + "days.npy"))
    benchmark_returns = np.load(os.path.join(datapath, "benchmark_returns_" + str(frequency) + "days.npy"))
    days = np.load(os.path.join(datapath, "todays_" + str(frequency) + "days.npy"))
    if pred_data_kind == "bool":
        fname = modelname + "_" + str(frequency) + "days_portfolios.npy"
    elif pred_data_kind == "proba":
        fname = modelname + "_" + str(frequency) + "days_portfolio_probas.npy"
    pred_data = np.load(os.path.join(savemodelpath, fname))

    bl = BlackLitterman(return_data, mean_cum_returns, market_data,
                        risk_free, pred_data, days, frequency,
                        pred_data_kind=pred_data_kind)
    weights = bl.get_weights(a=a)
    pngname = "BL_" + GAtype + "_" + modelname + "_" + str(frequency) + "days"
    # np.save(os.path.join(saveresultspath, pngname + ".npy"), weights)
    equitys = backtesting(weights, Y_portfolio, init_equity=init_equity)
    bench_equity = np.concatenate(([init_equity],benchmark_returns.cumprod() * init_equity))
    plot_backtest(equitys, bench_equity, fname = pngname, isshow = False)#, savepath = saveresultspath)

    days = pd.Series(np.concatenate((days, [20180629]))).apply(lambda x: pd.Timestamp.strptime(str(x), "%Y%m%d"))
    equitys = equity = pd.Series(equitys, index=days)
    bench_equity = pd.Series(bench_equity, index=days)
    results = get_results(equitys, bench_equity, frequency)
    print(a, "\t", results['tot_return'])

    # with open(os.path.join(saveresultspath, pngname + ".pkl"), "wb+") as file:
    #     pkl.dump(results, file)
    print("Done: ", pngname)

