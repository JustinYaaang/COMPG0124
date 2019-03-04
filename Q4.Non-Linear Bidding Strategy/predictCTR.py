import pandas as pd
import numpy as np


class predictCTR():
    def __init__(self, pClick):
        self._pClick = pClick

    def getPCTR(self):
        impr = self._pClick.shape[0]
        clicks = self._pClick['click'].sum()
        ctr = clicks*100/impr

        ratio = len(df_train) / 2 * np.bincount(df_train.click)
        balance_ratio = ratio[1] / ratio[0]

        new_pctrval = []
        new_pctrtest = []

        for pctr in pCTRval[1]:
            new_pctrval.append( pctr / (pctr + ((1-pctr) / balance_ratio)))

        for pctr in pCTRtest[1]:
            new_pctrtest.append( pctr / (pctr + ((1-pctr) / balance_ratio)))
        # pCTR = pd.DataFrame(self._pClick['click'])
        pCTR.to_csv('pCTR.csv')
        return pCTR
