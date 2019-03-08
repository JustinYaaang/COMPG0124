import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint    
import seaborn as sns
import datetime 

def EvalRandBidClicksOnly(dataframe, lowerBound, upperBound, budget, size):
    AdjustedBudget=(budget/size)*dataframe.shape[0]
    tempData=dataframe

    tempData['ConstBid'] = np.random.randint(lowerBound, upperBound+1, dataframe.shape[0])
    tempData['trueValues'] = np.where(tempData["payprice"]<tempData['ConstBid'],1,0)
    tempData['ModelPays'] = tempData['trueValues']*tempData['payprice']
    tempData['cumsum'] = tempData['ModelPays'].cumsum()
    lastRowToInclude = np.argmax(tempData['cumsum'].as_matrix()>AdjustedBudget)
    if lastRowToInclude==0:
        lastRowToInclude=tempData.shape[0]
    
    shortData = tempData.head(lastRowToInclude)
    trueValues = (0<shortData['ModelPays'])
    clicks = np.sum(shortData[trueValues]["click"].values)
    
    return clicks


def random_bidding_run(df, output):
    minBid=np.min(df["payprice"].values)
    maxBid=np.max(df["payprice"].values)

    print("minBid: {}".format(minBid))
    print("maxBid: {}".format(maxBid))


    minBidRange = list(range(0, 110, 10))
    click_list = []

    minValue, maxValue = [], []
    for minBid in minBidRange:
        for add_range in range(10, 310, 10):
            maxBid = minBid + add_range
            if maxBid > 300:
                break
            minValue.append(minBid)
            maxValue.append(maxBid)

            clicks = EvalRandBidClicksOnly(df, minBid, maxBid, budget, size)
            click_list.append(clicks)
            print("min: {}; max: {}; clicks: {}".format(minBid, maxBid, clicks))

    df = pd.DataFrame({'minValue': minValue, 'maxValue': maxValue, 'clicks': click_list})
    df.to_csv(output, encoding='utf-8', index=False)


if __name__ == "__main__":
    validation_df = pd.read_csv("../we_data/validation.csv")
    train_df = pd.read_csv("../we_data/train.csv")

    size = validation_df.shape[0]
    print("data fetched")
    budget=6250*1000

    # random_bidding_run(validation_df, 'random_bidding_validation.csv')
    random_bidding_run(train_df, 'random_bidding_train.csv')


