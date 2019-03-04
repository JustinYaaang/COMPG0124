import pandas as pd
import numpy as np


class predictCTR():
    def __init__(self, pClick):
        self._pClick = pClick

    def getPCTR(self):
        impr = self._pClick.shape[0]
        clicks = self._pClick['click'].sum()
        ctr = clicks*100/impr

        pCTR = pd.DataFrame()
        pCTR['ctr'] = ctr
        # pCTR = pd.DataFrame(self._pClick['click'])
        pCTR.to_csv('pCTR.csv')
        return pCTR
