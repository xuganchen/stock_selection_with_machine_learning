try:
    from .dataview import DataView
except:
    from dataview import DataView
import numpy as np
import pandas as pd
import os

BASICINFOR_NAME = "basic_infor"
FININDICATOR_NAME = "finIndicator"
TECHNICALINDEX_NAME = "technical_index"

def DataGenerate_WithQuantile(fpath,
                              quantile,
                              frequency):
    '''
    Generate training and testing data using in model

    :param fpath: the dictionary path with dataview data
    :param quantile:
    :param frequency:
    :return:
        X: (len, 299)
        Y: (len, )
    '''
    stocknumber = np.int(300 * quantile)

    basic_path = os.path.join(fpath, BASICINFOR_NAME)
    fin_path = os.path.join(fpath, FININDICATOR_NAME)
    tech_path = os.path.join(fpath, TECHNICALINDEX_NAME)


    # Loading Data

    ## loading factor dataview
    dv = DataView()
    dv.load_dataview(fpath)

    ## loading basic information dataview(including price and other information of HS300)
    dv_basic = DataView()
    dv_basic.load_dataview(basic_path)

    index_member = dv_basic.get_field("index_member")
    trade_status = dv_basic.get_field("trade_status")
    price = dv_basic.get_field("close_adj")
    price = price.loc[dv.start_date: dv.end_date]
    index_member = index_member.loc[dv.start_date: dv.end_date]
    trade_status = trade_status.loc[dv.start_date: dv.end_date]

    # Data Processing

    head_datas = None
    tail_datas = None
    for i in range(price.shape[0]):
        ## Step 1: get data
        today = price.index[i]
        today_status = trade_status.loc[today]
        print(today)
        today_stock = index_member.columns[index_member.iloc[i] == 1.0].values
        today_stock = np.array([stock for stock in today_stock if today_status[stock]])
        df = price.iloc[i:i+frequency][today_stock]

        if df.shape[0] != frequency:
            pass
        else:
            ## Step 2: replace nan with mean and data normalization
            data = dv.get_snapshot(today, symbol=today_stock.tolist())
            data = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
            data = pd.DataFrame(data = np.where(np.isnan(data),
                                                np.ma.array(data, mask=np.isnan(data)).mean(axis=0),
                                                data),
                                index = data.index, columns = data.columns)

            ## Step 3: calculate returns-to-std ratio
            returns = df.apply(lambda x: x.iloc[-1] / x.iloc[0], axis=0)
            std = df.apply(lambda x: np.std(x), axis=0)
            rs = (returns / std)
            rs = rs.loc[np.isfinite(rs)]
            rs = rs[(-rs).argsort()]

            ## Step 4: get head and tail data
            headstock = rs.iloc[:stocknumber].index.values
            tailstock = rs.iloc[-stocknumber:].index.values

            head_data = data.loc[headstock].values
            tail_data = data.loc[tailstock].values

            ## Step 5: store data
            if head_datas is None:
                head_datas = head_data
            else:
                head_datas = np.concatenate((head_datas, head_data))
            if tail_datas is None:
                tail_datas = tail_data
            else:
                tail_datas = np.concatenate((tail_datas, tail_data))

    # Shuffle Data
    X = np.concatenate((head_datas, tail_datas))
    Y = np.concatenate((np.ones(head_datas.shape[0]), np.zeros(tail_datas.shape[0])))
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation]
    Y = Y[permutation]

    return X, Y


def DataGenerate_WithoutQuantile(fpath,
                                 frequency):
    '''
    Generate protfolio data using in model

    :param fpath: the dictionary path with dataview data
    :param frequency:
    :return:
        Factors: (len, 300, 299)
        Prices: (len, 300, 3)
    '''

    basic_path = os.path.join(fpath, BASICINFOR_NAME)
    fin_path = os.path.join(fpath, FININDICATOR_NAME)
    tech_path = os.path.join(fpath, TECHNICALINDEX_NAME)

    # Loading Data

    ## loading factor dataview
    dv = DataView()
    dv.load_dataview(fpath)

    ## loading basic information dataview(including price and other information of HS300)
    dv_basic = DataView()
    dv_basic.load_dataview(basic_path)

    index_member = dv_basic.get_field("index_member")
    trade_status = dv_basic.get_field("trade_status")
    price = dv_basic.get_field("close_adj")
    price = price.loc[dv.start_date: dv.end_date]
    index_member = index_member.loc[dv.start_date: dv.end_date]
    trade_status = trade_status.loc[dv.start_date: dv.end_date]

    factors = []
    prices = []
    todays = []
    for i in range(price.shape[0]):
        if i % frequency == 0:
            ## Step 1: get data
            today = price.index[i]
            today_status = trade_status.loc[today]
            todays.append(today)
            print(today)
            today_stock = index_member.columns[index_member.iloc[i] == 1.0].values
            today_stock = np.array([stock for stock in today_stock if today_status[stock]])
            df = price.iloc[i:i+frequency][today_stock].values.T

            ## Step 2: replace nan with mean and data normalization
            data = dv.get_snapshot(today, symbol=today_stock.tolist())
            data = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
            data = np.where(np.isnan(data),
                            np.ma.array(data, mask=np.isnan(data)).mean(axis=0),
                            data)

            ## Step 3: store data
            factors.append(data)
            prices.append(df)
    factors = np.array(factors)
    prices = np.array(prices)
    todays = np.array(todays)

    return factors, prices, todays


def get_mean_cum_return(return_data, cum_freq = 60, start_date = 20121231):
    '''
    generate the mean of cummulativa return for each periods

    :param return_data: return data for everyday
    :param cum_freq: periods
    :param start_date:
    :return:
    '''
    start_date_index = np.where(return_data.index == start_date)[0][0]
    mean_cum_returns = pd.DataFrame(index=return_data.index[start_date_index:], columns=return_data.columns)
    for index in range(return_data.shape[0]):
        if index >= start_date_index:
            today = return_data.index[index]
            today_daily_return = return_data.iloc[index - cum_freq + 1:index + 1]
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
            mean_cum_returns.loc[today] = np.mean(stock_cum_returns)
            print(today)
    return mean_cum_returns


def get_benchmark_data(fpath, frequency = 10):
    '''
    calculate the benchmark price for every freq periods

    :param fpath:
    :param frequency:
    :param start_date:
    :return:
    '''

    basic_path = os.path.join(fpath, BASICINFOR_NAME)
    dv_basic = DataView()
    dv_basic.load_dataview(basic_path)
    data_benchmark = dv_basic.data_benchmark
    data_benchmark = data_benchmark.loc[dv_basic.start_date: dv_basic.end_date]

    data = []
    for i in range(data_benchmark.shape[0]):
        if i % frequency == 0:
            df = data_benchmark.iloc[i:i + frequency].values
            returns = df[-1] / df[0]
            data.append(returns)
    data = np.array(data)
    return data



if __name__ == '__main__':
    # fpath = "F:\\DeepLearning\\Data\\insample"
    # frequency = 5
    # quantile = 0.05
    #
    # X, Y = DataGenerate_WithQuantile(fpath, quantile, frequency)
    #
    # np.save(os.path.join(fpath, "X.npy"), X)
    # np.save(os.path.join(fpath, "Y.npy"), Y)



    # frequencyList = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30]
    # fpath = "F:\\DeepLearning\\Data\\outsample"
    #
    # for frequency in frequencyList:
    #     factors, prices, todays = DataGenerate_WithoutQuantile(fpath, frequency)
    #
    #     np.save(os.path.join(fpath, "factors_" + str(frequency) + "days.npy"), factors)
    #     np.save(os.path.join(fpath, "prices_" + str(frequency) + "days.npy"), prices)
    #     np.save(os.path.join(fpath, "todays_" + str(frequency) + "days.npy"), todays)
    #
    #     print("\n", frequency)
    #     print(factors.shape)
    #     print(factors[0].shape)
    #     print(prices.shape)
    #     print(prices[0].shape)
    #     print(todays.shape)



    filepath = "F:\\DeepLearning\\data\\outsample_total"
    return_data = pd.read_hdf(os.path.join(filepath, "return_data.h5"))
    mean_cum_returns = get_mean_cum_return(return_data)
    mean_cum_returns.to_hdf(os.path.join(filepath, "mean_cum_returns.h5"), key="mean_cum_returns")


    # fpath = "F:\\DeepLearning\\Data\\outsample"
    # frequencyList = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30]
    # for frequency in frequencyList:
    #     data = get_benchmark_data(fpath, frequency=frequency)
    #     np.save(os.path.join(fpath, "benchmark_returns_" + str(frequency) + "days.npy"), data)