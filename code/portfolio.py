import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def calcu_portfolio(model,
                    X_portfolio,
                    Y_portfolio,
                    savepath = None,
                    fname = None,
                    isshow = True,
                    init_equity = 100000):
    '''
    calculate the equity when we have the weight of portfolio.

    :param model: the model class
    :param X_portfolio: the test data, shape = (len, 300, 299)
    :param Y_portfolio: price series of every window, shape = (len, 300, 3)
    :param savepath: the dictionary path to save the figure
    :param fname: the name of figure
    :param isshow: True or False
    :param init_equity: initial equity
    :return:
    '''
    # Step 1: prediction
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

    # Step 2: backtesting
    portfolios_weight = []
    for i in range(len(portfolios)):
        portfolio = portfolios[i]
        portfolio_weight = portfolio / sum(portfolio)
        portfolios_weight.append(portfolio_weight)
    equity = backtesting(portfolios_weight, Y_portfolio, init_equity)

    # Step 3: plot
    fig = plt.figure(figsize=(10, 5))
    plt.plot(equity)
    plt.hlines(100000, 0, len(portfolios), color='black', linestyle='--')
    plt.ylabel("equity")
    if fname is None:
        plt.title(model.type)
    else:
        plt.title(fname)
    if isshow:
        plt.show()
    if savepath is not None:
        fig.savefig(os.path.join(savepath, fname))

    return equity

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
        portfolio_weight = portfolios_weight[i]
        prices = Y_portfolio[i]
        returns = prices[:, -1] / prices[:, 0]
        total = total * np.dot(portfolio_weight, returns)
        equity.append(total)
    return equity



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