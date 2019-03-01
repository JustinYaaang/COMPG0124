import pandas as pd
import numpy as np

def BuildCTRMatrix(dataset,plot):
    for n in range(0,repeats,1):
        datasubset = dataset.sample(frac=0.5, replace=False)

    return CTRMatrix


def FindBestBounds(ResultsMatrix):
    for x in minCustRange:
        for y in maxCustRange:
            if ResultsMatrix[x][y]==ResultsMatrix.values.max():
                print("X:",x,"Y:",y)
                lowerBound=x
                upperBound=y
    return lowerBound, upperBound

pd.set_option('display.max_columns', None)
train = pd.read_csv("../we_data/train.csv")

minBid=np.min(train["payprice"].values)
maxBid=np.max(train["payprice"].values)

custRange = np.arange(minBid, maxBid)

CTRMatrix =  BuildCTRMatrix(train,plot=False,repeats=5)
