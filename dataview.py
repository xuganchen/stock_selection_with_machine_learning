import os
import json
import errno
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class DataView(object):
    def __init__(self,
                 data_config = None):
        if data_config is None:
            self.data_config = {
                "remote.data.address": "tcp://data.quantos.org:8910",
                "remote.data.username": "18868116256",
                "remote.data.password": "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjI3NDE"
                                        "wNjkxMDAiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg4NjgxMTYyNTYif"
                                        "Q.9RSb6dJ_-a5mgrV9vCZsLxrCbuiZw4dGfwuWcP-AjEM"
            }
        else:
            self.data_config = data_config


        self.start_date = 20050101
        self.end_date = 20180701
        self.data_d = pd.DataFrame()
        self.data_d.columns = pd.MultiIndex.from_product([[], []])
        self.data_benchmark = pd.DataFrame()
        self.data_inst = pd.DataFrame()
        self.fields = []
        self.symbols = []
        self.universe = []
        self.benchmark = ''

        self.meta_data_list = ['start_date', 'end_date',
                               'fields', 'symbols', 'universe', 'benchmark']

    def _save_json(self, meta_data, meta_data_path):
        with open(meta_data_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, separators=(',\n', ': '))

    def _save_data(self, data, data_path):
        h5 = pd.HDFStore(data_path, complevel=9, complib='blosc')
        for key, value in data.items():
            h5[key] = value
        h5.close()

    def save_dataview(self, dataview_folder):
        abs_folder = os.path.abspath(dataview_folder)
        meta_data_path = os.path.join(abs_folder, 'meta_data.json')
        data_path = os.path.join(abs_folder, 'data.hd5')
        if not os.path.exists(abs_folder):
            os.makedirs(abs_folder)

        data_to_store = {
            'data_d': self.data_d,
            'data_benchmark': self.data_benchmark,
            'data_inst': self.data_inst
        }
        meta_data_to_store = {key: self.__dict__[key] for key in self.meta_data_list}

        self._save_json(meta_data_to_store, meta_data_path)
        self._save_data(data_to_store, data_path)

        print ("Dataview has been successfully saved to:\n" + abs_folder)

    def _load_json(self, meta_data_path):
        meta_data = dict()
        try:
            with open(meta_data_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        except:
            raise FileExistsError("There is no json file under file {}".format(meta_data_path))
        return meta_data

    def _load_data(self, data_path):
        h5 = pd.HDFStore(data_path)
        res = dict()
        for key in h5.keys():
            res[key] = h5.get(key)
        h5.close()
        return res

    def load_dataview(self, dataview_folder):
        meta_data_path = os.path.join(dataview_folder, "meta_data.json")
        data_path = os.path.join(dataview_folder, "data.hd5")
        if not(os.path.exists(meta_data_path) and os.path.exists(data_path)):
            raise FileExistsError("There is no data file under dictionary {}".format(dataview_folder))

        meta_data = self._load_json(meta_data_path)
        data = self._load_data(data_path)
        self.data_d = data.get("/data_d", None)
        self.data_benchmark = data.get("/data_benchmark", None)
        self.data_inst = data.get("/data_inst", None)
        self.__dict__.update(meta_data)

        print("Dataview loaded successfully.")

    def append_df(self, df, field_name):
        if isinstance(df, pd.DataFrame):
            pass
        elif isinstance(df, pd.Series):
            df = pd.DataFrame(df)
        else:
            raise ValueError("Data to be appended must be pandas format. But we have {}".format(type(df)))

        exist_symbols = self.data_d.columns.levels[0]
        if len(df.columns) < len(exist_symbols):
            df2 = pd.DataFrame(index=df.index, columns=exist_symbols, data=np.nan)
            df2.update(df)
            df = df2
        elif len(df.columns) > len(exist_symbols):
            df = df.loc[:, exist_symbols]
        multi_idx = pd.MultiIndex.from_product([exist_symbols, [field_name]])
        df.columns = multi_idx

        self.data_d = pd.merge(self.data_d, df, left_index=True, right_index=True, how='left')
        self.data_d = self.data_d.sort_index(axis=1)
        self.data_d.columns = self.data_d.columns.remove_unused_levels()

        self.fields.append(field_name)

    def append_df_symbol(self, df, symbol_name):
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            pass
        elif isinstance(df, pd.Series):
            df = pd.DataFrame(df)
        else:
            raise ValueError("Data to be appended must be pandas format. But we have {}".format(type(df)))

        exist_fields = self.data_d.columns.levels[1]
        if len(set(exist_fields) - set(df.columns)):
            df2 = pd.DataFrame(index=df.index, columns=exist_fields, data=np.nan)
            df2.update(df)
            df = df2
        multi_idx = pd.MultiIndex.from_product([[symbol_name], exist_fields])
        df.columns = multi_idx

        self.data_d = pd.merge(self.data_d, df, left_index=True, right_index=True, how='left')
        self.data_d = self.data_d.sort_index(axis=1)
        self.data_d.columns = self.data_d.columns.remove_unused_levels()

        self.symbols.append(symbol_name)

    def remove_field(self, field_names):
        if isinstance(field_names, str):
            field_names = field_names.split(',')
        elif isinstance(field_names, (list, tuple)):
            pass
        else:
            raise ValueError("field_names must be str or list of str.")

        for field_name in field_names:
            # parameter validation
            if field_name not in self.fields:
                print("Field name [{:s}] does not exist.".format(field_name))
                return
            # remove field data
            self.data_d = self.data_d.drop(field_name, axis=1, level=1)
            self.data_d.columns = self.data_d.columns.remove_unused_levels()
            # remove fields name from list
            self.fields.remove(field_name)

    def remove_symbols(self, symbols):
        if isinstance(symbols, str):
            symbols = symbols.split(',')
        elif isinstance(symbols, (list, tuple)):
            pass
        else:
            raise ValueError("symbols must be str or list of str.")

        for symbol in symbols:
            # parameter validation
            if symbol not in self.symbols:
                print("Field name [{:s}] does not exist.".format(symbol))
                return
            # remove field data
            self.data_d = self.data_d.drop(symbol, axis=1, level=0)
            self.data_d.columns = self.data_d.columns.remove_unused_levels()
            # remove fields name from list
            self.symbols.remove(symbol)


    def get(self, symbol=None, fields=None, start_date=None, end_date=None, data_format="wide"):
        sep = ","
        if fields is None:
            fields = slice(None)
        elif isinstance(fields, (list, tuple)):
            pass
        else:
            fields = fields.split(sep)
        if symbol is None:
            symbol = slice(None)
        elif isinstance(symbol, (list, tuple)):
            pass
        else:
            symbol = symbol.split(sep)
        if start_date is None:
            start_date = self.start_date
        if end_date  is None:
            end_date = self.end_date

        res = self.data_d.loc[pd.IndexSlice[start_date: end_date], pd.IndexSlice[symbol, fields]]
        if data_format is "wide":
            pass
        else:
            res = res.stack(level = "symbols").reset_index()
        return res

    def get_snapshot(self, snapshot_date, symbol=None, fields=None):
        res = self.get(symbol=symbol, fields=fields, start_date=snapshot_date, end_date=snapshot_date)
        if res is None:
            raise ValueError("No data for data={}, fields={}, symbol={}".format(snapshot_date, fields, symbol))

        res = res.stack(level="symbols", dropna=False)
        res.index = res.index.droplevel(level="trade_date")
        return res

    def get_symbol(self, symbol, fields=None, start_date=None, end_date=None):
        res = self.get(symbol=symbol, fields=fields, start_date=start_date, end_date=end_date)
        if res is None:
            raise ValueError("No data. for "
                             "start_date={}, end_date={}, field={}, symbol={}".format(start_date, end_date,
                                                                                      fields, symbol))

        res.columns = res.columns.droplevel(level='symbols')
        return res

    def get_field(self, field, symbol=None, start_date=None, end_date=None):
        res = self.get(symbol=symbol, fields=field, start_date=start_date, end_date=end_date)
        if res is None:
            raise ValueError("No data. for "
                             "start_date={}, end_date={}, field={}, symbol={}".format(start_date, end_date,
                                                                                      field, symbol))

        res.columns = res.columns.droplevel(level='fields')
        return res

    def reflash_data(self, start_date=None, end_date=None, fields=None, symbol=None):
        sep = ","
        if fields is None:
            fields = slice(None)
        else:
            fields = fields.split(sep)
        if symbol is None:
            symbol = slice(None)
        else:
            symbol = symbol.split(sep)
        if start_date is None:
            start_date = self.start_date
        if end_date  is None:
            end_date = self.end_date

        self.start_date = start_date
        self.end_date = end_date
        self.data_d = self.data_d.loc[pd.IndexSlice[start_date: end_date], pd.IndexSlice[symbol, fields]]
        self.data_d.columns = self.data_d.columns.remove_unused_levels()
        self.data_benchmark = self.data_benchmark.loc[pd.IndexSlice[start_date: end_date]]


