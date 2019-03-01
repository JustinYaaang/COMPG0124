import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint    
import seaborn as sns
import datetime 

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


def BuildCTRMatrix(dataframe,plot,repeats):
    #lastInRange = custRange[len(custRange)-1:][0]+1
    for n in range(0,repeats,1):
        dataSubset = dataframe.sample(frac=0.5, replace=False)
        print(n+1,"/",repeats,"...")
        
        for minimumBid in minCustRange:
            print('counting clicks',minimumBid,datetime.datetime.now().time())        
            for maximumBid in maxCustRange: #maxBid>minBid?????
                if maximumBid>minimumBid:
                    Clicks = EvalRandBidClicksOnly(dataSubset,minimumBid, maximumBid,6250000,validation_df.shape[0])
                    CTRMatrix[minimumBid][maximumBid]=(CTRMatrix[minimumBid][maximumBid]*n+Clicks)/(n+1)
        CTRMatrix.to_csv("RandomBidResults.csv")
    return CTRMatrix


def FindBestBounds(ResultsMatrix):
    print("Best CTR is:",ResultsMatrix.values.max())
    for x in minCustRange:
        for y in maxCustRange:
            if ResultsMatrix[x][y]==ResultsMatrix.values.max():
                print("X:",x,"Y:",y)
                lowerBound=x
                upperBound=y
    return lowerBound, upperBound



def EvalRandBid(dataframe,lowerBound, upperBound,budget,size):
    
    AdjustedBudget=(budget/size)*dataframe.shape[0]
    tempData=dataframe
    tempData['ConstBid'] = np.random.randint(lowerBound, upperBound+1, dataframe.shape[0])
    tempData['trueValues'] = np.where(tempData["payprice"]<tempData['ConstBid'],1,0)
    tempData['ModelPays'] = tempData['trueValues']*tempData['payprice']
    tempData['cumsum'] = tempData['ModelPays'].cumsum()
    lastRowToInclude = np.argmax(tempData['cumsum'].as_matrix()>AdjustedBudget)
    if lastRowToInclude==0:
        lastRowToInclude=tempData.shape[0]
    #print("lastRowToInclude",lastRowToInclude)
    
    shortData = tempData.head(lastRowToInclude).copy(True)
    trueValues = (0<shortData['ModelPays'])
    impressions = shortData[trueValues].shape[0]
    clicks = np.sum(shortData[trueValues]["click"].values)
    spend = np.sum(shortData[trueValues]["payprice"].values)
    CostPerClick = spend/clicks
    CostPerMille = spend*1000/(impressions)
    ClickThroughRate=(clicks*100)/impressions
    #print(constant, ClickThroughRate, clicks, spend, CostPerMille, CostPerClick, impressions)
    return [ClickThroughRate, clicks, spend, CostPerMille, CostPerClick, impressions]


def EvalRandBidClicksOnly(dataframe,lowerBound, upperBound,budget,size):
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


def multi_random(dataframe, n, lowerBound, upperBound, budget, size):
    AdjustedBudget=(budget/size)*dataframe.shape[0]
    tempData=dataframe
    print(tempData.shape)
    price_array, clicks_array = [], []

    for i in range(n):
        price_array.append(np.random.randint(lowerBound, upperBound+1, dataframe.shape[0]))

    # print(clicks_array)

    for i in range(n):
        tempData['trueValues'] = np.where(tempData["payprice"]<price_array[i],1,0)
        
        # for j in range(tempData['trueValues'].shape[0]):
        #     max_value = price_array[i].max()
        #     count = 0
        #     index = 0
        #     for m in range(n):
        #         if price_array[m][j] == max_value:
        #             count += 1
        #             index = m
        #         if count > 1:
        #             break
        #     if count > 1:
        #         tempData['trueValues'][j] = 0
        #     if count == 1 and index != j:
        #         tempData['trueValues'][j] = 0

    #     tempData['ModelPays'] = tempData['trueValues']*tempData['payprice']
    #     tempData['cumsum'] = tempData['ModelPays'].cumsum()
    #     lastRowToInclude = np.argmax(tempData['cumsum'].as_matrix()>AdjustedBudget)
    #     if lastRowToInclude==0:
    #         lastRowToInclude=tempData.shape[0]
        
    #     shortData = tempData.head(lastRowToInclude)
    #     trueValues = (0<shortData['ModelPays'])
    #     clicks = np.sum(shortData[trueValues]["click"].values)
    #     clicks_array.append(clicks)
    # print(clicks_array)
    # return clicks_array



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


# random_bidding_run()
multi_random(validation_df, 50, 30, 110, budget, validation_df.shape[0])
