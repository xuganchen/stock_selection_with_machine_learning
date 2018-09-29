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
    price = dv_basic.get_field("close_adj")
    index_member = index_member.loc[dv.start_date: dv.end_date]
    price = price.loc[dv.start_date: dv.end_date]


    # Data Processing

    head_datas = None
    tail_datas = None
    for i in range(price.shape[0]):
        ## Step 1: get data
        today = price.index[i]
        print(today)
        today_stock = index_member.columns[index_member.iloc[i] == 1.0].values
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
    price = dv_basic.get_field("close_adj")
    index_member = index_member.loc[dv.start_date: dv.end_date]
    price = price.loc[dv.start_date: dv.end_date]

    factors = []
    prices = []
    for i in range(price.shape[0]):
        if i % frequency == 0:
            ## Step 1: get data
            today = price.index[i]
            print(today)
            today_stock = index_member.columns[index_member.iloc[i] == 1.0].values
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
    # prices = prices

    return factors, prices





if __name__ == '__main__':
    # fpath = "F:\\DeepLearning\\Data\\insample"
    # frequency = 3
    # quantile = 0.05
    #
    # X, Y = DataGenerate_WithQuantile(fpath, quantile, frequency)
    #
    # np.save(os.path.join(fpath, "X.npy"), X)
    # np.save(os.path.join(fpath, "Y.npy"), Y)


    fpath = "F:\\DeepLearning\\Data\\insample"
    frequency = 5

    factors, prices = DataGenerate_WithoutQuantile(fpath, frequency)

    np.save(os.path.join(fpath, "factors_" + str(frequency) + "days.npy"), factors)
    np.save(os.path.join(fpath, "prices_" + str(frequency) + "days.npy"), prices)

    print(factors.shape)
    print(factors[0].shape)
    print(prices.shape)
    print(prices[0].shape)