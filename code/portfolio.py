import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from itertools import groupby


def get_avg_weight(pred_data):
    '''
    calculate weights using average weight method from pred_data

    :param pred_data: the array of portfolio, buy or not buy, 1/0,
                                and the shape is (len, 300)
    :return: the weights of portfolio
    '''
    weights = []
    for i in range(pred_data.shape[0]):
        pred = pred_data[i]
        total = sum(pred > 0)
        weight = pred / total
        weights.append(weight)
    return np.array(weights)


def prediction(model,
              X_portfolio,
              savepath = None,
              fname = None):
    '''
    prediction model

    :param model: the model class
    :param X_portfolio: the test data, shape = (len, 300, 299)
    :param savepath: the dictionary path to save the figure
    :param fname: the name of figure
    :return:
    '''
    portfolios = []
    portfolio_probas = []
    for i in range(len(X_portfolio)):
        factors = X_portfolio[i]
        portfolio = model.predict(factors)
        portfolio_proba = model.predict_proba(factors)
        portfolios.append(portfolio)
        portfolio_probas.append(portfolio_proba)
    portfolios = np.array(portfolios)
    portfolio_probas = np.array(portfolio_probas)
    if savepath is not None:
        np.save(os.path.join(savepath, fname + "_portfolios.npy"), portfolios)
        np.save(os.path.join(savepath, fname + "_portfolio_probas.npy"), portfolio_probas)
    return portfolios, portfolio_probas

def backtesting(portfolios_weight,
                Y_portfolio,
                init_equity = 100000):
    '''
    backtesting

    :param portfolios_weight: the array of portfolio, buy or not buy, 1/0,
                                and the shape is (len, 300)
    :param Y_portfolio: the price of every stocks for everyday
    :param init_equity: initial euiqty
    :return:
    '''
    total = init_equity
    equity = [total]
    for i in range(len(portfolios_weight)):
        portfolio_weight = portfolios_weight[i].reshape(-1)
        prices = Y_portfolio[i]
        returns = prices[:, -1] / prices[:, 0]
        total = total * np.dot(portfolio_weight, returns)
        equity.append(total)
    return np.array(equity)

def plot_backtest(equity, bench_equity, fname = None, savepath = None, isshow = None):
    fig = plt.figure(figsize=(10, 5))
    returns = equity[-1] / equity[0] - 1
    returns_ben = bench_equity[-1] / bench_equity[0] - 1
    label = fname + ':  {:.3%}'.format(returns)
    label_ben = "HS300" + ':  {:.3%}'.format(returns_ben)
    plt.plot(equity, label=str(label))
    plt.plot(bench_equity, label=str(label_ben))
    plt.hlines(100000, 0, len(equity), color='black', linestyle='--')
    plt.ylabel("equity")
    plt.legend()
    if fname is not None:
        plt.title(fname)
    if isshow:
        plt.show()
    if savepath is not None:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        fig.savefig(os.path.join(savepath, fname))


def plot_frequency(equitys,
                   fname,
                   savepath=None,
                   isshow=True):
    '''
    plot the frequency model under different model

    :param equitys: the dict of equity, dict = {model: equity series}
    :param savepath: the dictionary path to save the figure
    :param fname: the name of figure
    :param isshow: True or False
    :return:
    '''
    leng = 0
    for name in equitys:
        leng = max(leng, len(equitys[name]))
    new_equitys = {}
    for name in equitys:
        new_equitys[name] = _extend_equity(equitys[name], leng)

    fig = plt.figure(figsize=(10, 5))
    for name in new_equitys:
        equity = new_equitys[name]
        returns = equity.iloc[-1] / equity.iloc[0] - 1
        label = name + '   ' + '{:.3%}'.format(returns)
        equity.plot(label=str(label))
    plt.hlines(100000, 0, leng, color='black', linestyle='--')
    plt.title(fname)
    plt.legend()
    plt.show()

    if isshow:
        plt.show()
    if savepath is not None:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        fig.savefig(os.path.join(savepath, fname))


def _extend_equity(equity, leng):
    new_equity = pd.Series([np.nan for _ in range(leng)])
    new_equity[0] = equity[0]
    new_equity[leng] = equity[-1]
    pos = np.concatenate((np.ones(len(equity) - 2), np.zeros(leng - len(equity) + 2)))
    permutation = np.random.permutation(leng)
    pos = pos[permutation]
    pos = np.concatenate(([0], pos, [0]))
    new_equity.loc[pos == 1] = equity[1:-1]
    new_equity = pd.Series.interpolate(new_equity)
    return new_equity


def plot_model(equitys,
               fname,
               savepath=None,
               isshow=True):
    '''
    plot the same model under different frequency

    :param equitys: the dict of equity, dict = {freq: equity series}
    :param savepath: the dictionary path to save the figure
    :param fname: the name of figure
    :param isshow: True or False
    :return:
    '''

    fig = plt.figure(figsize=(10, 5))
    for name in equitys:
        equity = pd.Series(equitys[name])
        returns = equity.iloc[-1] / equity.iloc[0] - 1
        label = name + '   ' + '{:.3%}'.format(returns)
        equity.plot(label=str(label))
    plt.hlines(100000, 0, len(equity), color='black', linestyle='--')
    plt.title(fname)
    plt.legend()
    plt.show()

    if isshow:
        plt.show()
    if savepath is not None:
        fig.savefig(os.path.join(savepath, fname))

def _create_drawdown(cum_returns):
    idx = cum_returns.index
    hwm = cum_returns.expanding(min_periods=1).max()
    dd = pd.DataFrame(index = idx)
    dd['Drawdown'] = (hwm - cum_returns) / hwm
    dd['Drawdown'].iloc[0] = 0.0
    dd['Duration'] = np.where(dd['Drawdown'] == 0, 0, 1)
    duration = max(sum(g) for k,g in groupby(dd['Duration']))
    return dd['Drawdown'], np.max(dd['Drawdown']), duration



def get_results(equity, bench_equity, frequency, periods = 250):
    res_equity = pd.Series(equity).sort_index()

    # Returns
    res_returns = res_equity.pct_change().fillna(0.0)

    # Cummulative Returns
    res_cum_returns = res_equity / equity.iloc[0]

    # totalreturn
    res_tot_return = res_cum_returns.iloc[-1] - 1.0

    # annualized rate of return
    times = res_equity.index
    years = (times[-1] - times[0]).days / (365)
    res_annual_return = res_tot_return / years
    res_cagr = (res_cum_returns.iloc[-1] ** (1.0 / years)) - 1.0

    # Drawdown, max drawdown, max drawdown duration
    res_drawdown, res_max_dd, res_mdd_dur = _create_drawdown(res_cum_returns)

    # Sharpe Ratio
    if np.std(res_returns) == 0:
        res_sharpe = np.nan
    else:
        res_sharpe = np.sqrt(periods) * np.mean(res_returns) / np.std(res_returns)

    # sortino ratio
    if np.std(res_returns[res_returns < 0]) == 0:
        res_sortino = np.nan
    else:
        res_sortino = np.sqrt(periods) * (np.mean(res_returns)) / np.std(res_returns[res_returns < 0])

    # BNH
    res_bench_equity = pd.Series(bench_equity).sort_index()
    res_bench_returns = res_bench_equity.pct_change().fillna(0.0)
    res_bench_cum_returns = res_bench_returns / res_bench_equity.iloc[0]
    IR_returns = res_returns - res_bench_returns
    if np.std(IR_returns) == 0:
        res_IR = np.nan
    else:
        res_IR = np.sqrt(periods) * np.mean(IR_returns) / np.std(IR_returns)

    # rolling return
    ## by Year
    res_rolling_return_year = res_equity.resample("Y").apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    res_rolling_return_year.index = res_rolling_return_year.index.year

    results = {}
    results['equity'] = res_equity
    results['returns'] = res_returns
    results['cum_returns'] = res_cum_returns
    results['tot_return'] = res_tot_return
    results['annual_return'] = res_annual_return
    results['cagr'] = res_cagr
    results['drawdown'] = res_drawdown
    results['max_drawdown'] = res_max_dd
    results['max_drawdown_duration'] = res_mdd_dur
    results['sharpe'] = res_sharpe
    results['sortino'] = res_sortino
    results['IR'] = res_IR
    results['rolling_return_year'] = res_rolling_return_year
    results['bench_equity'] = res_bench_equity
    results['bench_returns'] = res_bench_returns
    results['bench_cum_returns'] = res_bench_cum_returns

    return results
