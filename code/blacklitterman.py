try:
    from .dataview import DataView
except:
    from dataview import DataView
import numpy as np
import pandas as pd
import os
from cvxopt import solvers, matrix

BASICINFOR_NAME = "basic_infor"
FININDICATOR_NAME = "finIndicator"
TECHNICALINDEX_NAME = "technical_index"


class BlackLitterman(object):
    def __init__(self,
                 return_data,
                 mean_cum_returns,
                 market_data,
                 risk_free,
                 pred_data,
                 days,
                 freq,
                 pred_data_kind = "proba",
                 cum_freq = 60,
                 tau = 0.01,
                 delta = 0.5,
                 fpath = None
                 ):
        self.return_data = return_data
        self.mean_cum_returns = mean_cum_returns
        self.market_data = market_data
        self.risk_free = risk_free
        self.pred_data = pred_data
        self.days = days
        self.freq = freq
        if (pred_data_kind != "proba") and (pred_data_kind != "bool"):
            raise ValueError("The pred_data_kind should be 'proba' or 'bool'")
        else:
            self.pred_data_kind = pred_data_kind


        self.cum_freq = cum_freq
        self.tau = tau
        self.delta = delta

        if fpath is None:
            fpath = "F:\\DeepLearning\\data\\outsample"
        self.basic_path = os.path.join(fpath, BASICINFOR_NAME)
        self.fin_path = os.path.join(fpath, FININDICATOR_NAME)
        self.tech_path = os.path.join(fpath, TECHNICALINDEX_NAME)

        self.index_member, self.trade_status, self.start_date, self.end_date = self._init_dataview()

    def _init_dataview(self):
        dv_basic = DataView()
        dv_basic.load_dataview(self.basic_path)
        start_date = dv_basic.start_date
        end_date = dv_basic.end_date
        index_member = dv_basic.get_field("index_member")
        trade_status = dv_basic.get_field("trade_status")
        index_member = index_member.loc[start_date: end_date]
        trade_status = trade_status.loc[start_date: end_date]
        index_member.columns = pd.Series(index_member.columns).apply(lambda x: x[:-3])
        trade_status.columns = pd.Series(trade_status.columns).apply(lambda x: x[:-3])
        return index_member, trade_status, start_date, end_date

    def get_pi(self, today_stock, today_risk_free, today_mkt_return, today_daily_return):
        mean_mkt_return = np.mean(today_mkt_return)
        var_mkt_return = np.var(today_mkt_return)

        betas = []
        for stock in today_stock:
            beta = np.cov(today_daily_return[stock], today_mkt_return)
            betas.append(beta[1, 0])
        betas = np.array(betas) / var_mkt_return

        pi = betas * (mean_mkt_return - today_risk_free.values[0]) + today_risk_free.values[0]

        return np.matrix(pi).T

    def get_cov(self, today_daily_return):
        cov = np.cov(today_daily_return, rowvar=False)
        return np.matrix(cov)

    def get_q_P_Omega(self, pred_data, today_mean_cum_returns):
        if self.pred_data_kind == "proba":
            rank = pred_data.argsort().argsort()
            threshold = len(rank) // 2
            P = rank
            P[rank >= threshold] = 1
            P[rank < threshold] = -1
        elif self.pred_data_kind == "bool":
            P = pred_data
            P[P == 0] = -1

        qqq = P * today_mean_cum_returns

        q = np.mean(qqq)
        Omega = np.var(qqq)
        return np.matrix(q), np.matrix(P), np.matrix(Omega)


    def get_weights(self):
        weights = []
        for index in range(len(self.days)):
            today = self.days[index]
            today_status = self.trade_status.loc[today]
            today_stock = self.index_member.columns[self.index_member.loc[today] == 1.0].values
            today_stock = np.array([stock for stock in today_stock if today_status[stock]])

            i = np.where(self.return_data.index == today)[0][0]
            today_daily_return = self.return_data.iloc[i - self.cum_freq + 1:i + 1][today_stock]
            today_mean_cum_returns = self.mean_cum_returns.loc[today, today_stock]
            today_mkt_return = self.market_data.iloc[i - self.cum_freq + 1:i + 1]['market_data']
            today_risk_free = self.risk_free.loc[today]
            pred_data = self.pred_data[index]

            cov = self.get_cov(today_daily_return)
            pi = self.get_pi(today_stock, today_risk_free, today_mkt_return, today_daily_return)
            q, P, Omega = self.get_q_P_Omega(pred_data, today_mean_cum_returns)

            # miu
            first_part = ((self.tau * cov).I + np.matrix(np.dot(np.dot(P.T, 1/Omega), P))).I
            second_part = np.dot((self.tau * cov).I, pi) + np.dot(np.dot(P.T, 1/Omega), q)
            miu = np.dot(first_part, second_part)

            # V
            V = first_part

            # # weight
            # w = np.dot((self.delta * V).I, miu)
            # w = w / sum(w)
            # weights.append(np.array(w).reshape(-1))
            # # print(today)


            # optimazation
            PP = matrix(self.delta * V)
            qq = matrix(-1 * pi)
            GG = matrix(np.concatenate((-1 * np.eye(len(pi)), np.eye(len(pi)))))
            hh = matrix(np.concatenate((0.02 * np.ones(len(pi)), 0.02 * np.ones(len(pi)))))
            # hh = matrix(np.concatenate((np.zeros(len(pi)), 0.02 * np.ones(len(pi)))))
            AA = matrix(np.ones(len(pi)).reshape(1, -1))
            bb = matrix([1.0])

            solvers.options['show_progress'] = False
            sol = solvers.qp(PP, qq, GG, hh, AA, bb)
            w = np.array(sol['x'])
            weights.append(w.reshape(-1))

        return weights


if __name__ == '__main__':
    frequency = 5
    model = "DNN"
    filepath = "F:\\DeepLearning\\data\\outsample_total"

    pred_datapath = "F:\\DeepLearning\\Model\\20181002-012155\\GA_before\\prediction\\" + model
    name = model + "_" + str(frequency) + "days_portfolio_probas.npy"

    pred_data = np.load(os.path.join(ffpath, name))


    days = np.load(os.path.join(ffpath, "todays_5days.npy"))
    return_data = pd.read_hdf(os.path.join(filepath, "return_data.h5"))
    mean_cum_returns = pd.read_hdf(os.path.join(filepath, "mean_cum_returns.h5"))
    market_data = pd.read_hdf(os.path.join(filepath, "market_data.h5"))
    risk_free = pd.read_hdf(os.path.join(filepath, "risk_free.h5"))

    bl = BlackLitterman(return_data, mean_cum_returns, market_data, risk_free, pred_data, days, frequency)
    weights = bl.get_weights()