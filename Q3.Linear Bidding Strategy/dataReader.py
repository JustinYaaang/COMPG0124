import pandas as pd
import numpy as np


class dataReader():

    def __init__(self, filename):
        print(filename)
        pd.set_option('display.max_columns', None)
        self.__dataframe = pd.read_csv(filename, header=0)

    def getDataFrame(self):
        return self.__dataframe

    def getOneHotData(self, train_cols=[], exclude_domain=True, domain_keep_prob=0.05):
        sourceDF = self.__dataframe.copy()

        sourceDF['user_platform'] = sourceDF.useragent.str.split(
            '_').str.get(0)
        sourceDF['user_browser'] = sourceDF.useragent.str.split(
            '_').str.get(1)

        onehotDF = pd.get_dummies(sourceDF, columns=[
                                                     'weekday', 'hour',
                                                     'user_platform', 'user_browser',
                                                     'region', 'city',
                                                     # 'adexchange', 'domain',
                                                     # 'slotwidth', 'slotheight', 'slotvisibility',
                                                     # 'slotformat', 'creative', 'keypage', 'advertiser',
                                                     ])
        return onehotDF
