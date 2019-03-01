import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint    
import seaborn as sns
import datetime 




def plotResults(Matrix):

    ax = sns.heatmap(Matrix, linewidth=0, xticklabels=minCustRange, yticklabels=maxCustRange[::-1], cmap="Greens") #, annot=True
    ax.set(xlabel='lower bound for random bid', ylabel='upper bound for random bid', title="Clicks by bounded random bids")
    for label in ax.xaxis.get_ticklabels()[::1]:
        label.set_visible(False)
    for label in ax.xaxis.get_ticklabels()[::5]:
        label.set_visible(True)
    for label in ax.yaxis.get_ticklabels()[::1]:
        label.set_visible(False)
    for label in ax.yaxis.get_ticklabels()[::5]:
        label.set_visible(True)
        
    plt.savefig('RandomBidResults.png')
    plt.show()
    
    useless = 0
    return useless








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




def random_bidding_run():
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

            clicks = EvalRandBidClicksOnly(validation_df, minBid, maxBid, budget,validation_df.shape[0])
            click_list.append(clicks)
            print("min: {}; max: {}; clicks: {}".format(minBid, maxBid, clicks))

    df = pd.DataFrame({'minValue': minValue, 'maxValue': maxValue, 'clicks': click_list})
    df.to_csv('random_bidding_clicks.csv', encoding='utf-8', index=False)




if __name__ == "__main__":
    validation_df = pd.read_csv("../we_data/validation.csv")

    print("data fetched")
    budget=6250*1000

    minBid=np.min(validation_df["payprice"].values)
    maxBid=np.max(validation_df["payprice"].values)

    print("minBid: {}".format(minBid))
    print("maxBid: {}".format(maxBid))

    step_size = 1
    custRange = np.arange(minBid+1, maxBid+5,step_size) # determines the range that bids should be in

    minCustRange = np.arange(21, 91, step_size) # determines the range that bids should be in
    maxCustRange = np.arange(71, 161, step_size) # determines the range that bids should be in

    pd.set_option('display.max_columns', None)
