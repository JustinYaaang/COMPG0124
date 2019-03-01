import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint    
import seaborn as sns
import datetime 


validation_df = pd.read_csv("../we_data/validation.csv")
# train_df = pd.read_csv("../we_data/train.csv")

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
        
#     for label in ax.xaxis.get_ticklabels()[::2]:
#         label.set_visible(False)    
#     for label in ax.yaxis.get_ticklabels()[::2]:
#         label.set_visible(False)    
        
        
    plt.savefig('RandomBidResults.png')
    plt.show()
    
    useless = 0
    return useless



def EvalBidClicksOnly(dataframe, bidprice,budget,size):
    
    AdjustedBudget=(budget/size)*dataframe.shape[0]
    #print("AdjustedBudget is:",AdjustedBudget)
    tempData=dataframe
    #tempData['ConstBid']=constant
    tempData['ConstBid'] = bidprice

    tempData['trueValues'] = np.where(tempData["payprice"]<tempData['ConstBid'],1,0)
    tempData['ModelPays'] = tempData['trueValues']*tempData['payprice']
    tempData['cumsum'] = tempData['ModelPays'].cumsum()
    lastRowToInclude = np.argmax(tempData['cumsum'].as_matrix()>AdjustedBudget)
    if lastRowToInclude==0:
        lastRowToInclude=tempData.shape[0]
    #print("lastRowToInclude",lastRowToInclude)
    
    shortData = tempData.head(lastRowToInclude)
    trueValues = (0<shortData['ModelPays'])
    ##impressions = shortData[trueValues].shape[0]
    clicks = np.sum(shortData[trueValues]["click"].values)
    return clicks


    #impressions = shortData.loc[shortData['ModelPays'] > 0].sum()
    #clicks = shortData.loc[shortData['ModelPays'] > 0 , 'click'].sum()
    #print("clicks:",clicks)


def constant_bidding():
    click_list = []

    for i in range(1, maxBid + 1):
        clicks = EvalBidClicksOnly(validation_df, i, budget,validation_df.shape[0])
        click_list.append(clicks)
        print("bidding_price: {}; clicks: {}".format(i, clicks))

    df = pd.DataFrame({'bidding_price': list(range(1, maxBid + 1)), 'clicks': click_list})
    df.to_csv('constant_bidding_clicks.csv', encoding='utf-8', index=False)


constant_bidding()











