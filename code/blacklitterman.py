try:
    from .dataview import DataView
except ModuleNotFoundError:
    from dataview import DataView
import numpy as np
import pandas as pd
import os

BASICINFOR_NAME = "basic_infor"
FININDICATOR_NAME = "finIndicator"
TECHNICALINDEX_NAME = "technical_index"


class BlackLitterman(object):
    def __init__(self,
                 return_data,
                 market_data,
                 risk_free,
                 pred_data,
                 freq = 60,
                 tau = 0.01,
                 delta = 0.5,
                 fpath = None
                 ):
        self.return_data = return_data
        self.market_data = market_data
        self.risk_free = risk_free
        self.pred_data = pred_data
        self.freq = freq
        self.start_date = 20141231
        self.end_date = 20180701

        self.tau = tau
        self.delta = delta

        if fpath is None:
            fpath = "F:\\DeepLearning\\data\\outsample"
        self.basic_path = os.path.join(fpath, BASICINFOR_NAME)
        self.fin_path = os.path.join(fpath, FININDICATOR_NAME)
        self.tech_path = os.path.join(fpath, TECHNICALINDEX_NAME)

        self.index_member = self._init_dataview()

    def _init_dataview(self):
        dv_basic = DataView()
        dv_basic.load_dataview(self.basic_path)
        index_member = dv_basic.get_field("index_member")
        index_member = index_member.loc[self.start_date: self.end_date]
        index_member.columns = pd.Series(index_member.columns).apply(lambda x: x[:-3])
        return index_member

    def get_pi(self, today_stock, today_risk_free, today_mkt_return, today_daily_return):
        mean_mkt_return = np.mean(today_mkt_return)
        var_mkt_return = np.var(today_mkt_return)

        betas = []
        for stock in today_stock:
            beta = np.cov(today_daily_return[stock], today_mkt_return)
            betas.append(beta[1, 0])
        betas = np.array(betas) / var_mkt_return

        pi = betas * (mean_mkt_return - today_risk_free).values

        return np.matrix(pi)

    def get_cov(self, today_daily_return):
        cov = np.cov(today_daily_return, rowvar=False)
        return np.matrix(cov)

    def get_q_P_Omega(self, pred_data):

        P = None

        stock_cum_returns = {}
        for stock in today_daily_return.columns:
            stock_return = today_daily_return[stock]
            stock_cum_return = stock_return.rolling(10).apply(lambda x: (x + 1).cumprod()[-1] - 1)
            stock_cum_return_done = []
            for i in range(len(stock_cum_return)):
                if i % 10 == 9:
                    stock_cum_return_done.append(stock_cum_return.iloc[i])
            stock_cum_returns[stock] = stock_cum_return_done
        stock_cum_returns = pd.DataFrame(stock_cum_returns, columns=stock_cum_returns.keys())
        mean_cum_return = np.mean(stock_cum_returns).values
        qqq = P * mean_cum_return

        q = np.mean(qqq)
        Omega = np.var(qqq)
        return np.matrix(q), np.matrix(P), np.matrix(Omega)


    def get_weigh(self):
        today_stock = self.index_member.columns[self.index_member.loc[today] == 1.0].values
        i = np.where(self.return_data.index == today)[0][0]
        today_daily_return = self.return_data.iloc[i - freq + 1:i + 1][today_stock]
        today_mkt_return = self.market_data.iloc[i - freq + 1:i + 1]['mkt_return']
        today_risk_free = self.risk_free.loc[today]

        cov = self.get_cov(today_daily_return)
        pi = self.get_pi(today_stock, today_risk_free, today_mkt_return, today_daily_return)
        q, P, Omega = self.get_q_P_Omega(pred_data)

        # miu
        first_part = ((self.tau * cov).I + np.dot(np.dot(P.T, Omega.I), P)).I
        second_part = np.dot((self.tau * cov).I, pi) + np.dot(np.dot(P.T, 1/Omega), q)
        miu = np.dot(first_part, second_part)

        # V
        V = first_part

        # weight
        w = np.dot(miu, self.delta * V)

        return w