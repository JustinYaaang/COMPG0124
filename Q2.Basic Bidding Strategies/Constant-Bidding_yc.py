import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import seaborn as sns
import datetime


validation_df = pd.read_csv("../we_data/validation.csv")
train_df = pd.read_csv("../we_data/train.csv")
size = validation_df.shape[0]

print("data fetched")
budget=6250*1000

minBid=np.min(validation_df["payprice"].values)
# minBid=20

maxBid=np.max(validation_df["payprice"].values)

print("minBid: {}".format(minBid))
print("maxBid: {}".format(maxBid))

step_size = 1
custRange = np.arange(minBid+1, maxBid+5,step_size) # determines the range that bids should be in

minCustRange = np.arange(21, 91, step_size) # determines the range that bids should be in
maxCustRange = np.arange(71, 161, step_size) # determines the range that bids should be in

pd.set_option('display.max_columns', None)


def EvalBidClicksOnly(dataframe, bidprice,budget,size):

    AdjustedBudget=(budget/size)*dataframe.shape[0]
    #print("AdjustedBudget is:",AdjustedBudget)
    tempData=dataframe
    #tempData['ConstBid']=constant
    tempData['ConstBid'] = bidprice

    tempData['trueValues'] = np.where(tempData["payprice"]<=tempData['ConstBid'],1,0)
    tempData['ModelPays'] = tempData['trueValues']*tempData['payprice']
    tempData['cumsum'] = tempData['ModelPays'].cumsum()
    lastRowToInclude = np.argmax(tempData['cumsum'].as_matrix()>AdjustedBudget)
    if lastRowToInclude==0:
        lastRowToInclude=tempData.shape[0]
    #print("lastRowToInclude",lastRowToInclude)

    shortData = tempData.head(lastRowToInclude)
    trueValues = (0<shortData['ModelPays'])
    impressions = shortData[trueValues].shape[0]
    clicks = np.sum(shortData[trueValues]["click"].values)
    return clicks


    # impressions = shortData.loc[shortData['ModelPays'] > 0].sum()
    # clicks = shortData.loc[shortData['ModelPays'] > 0 , 'click'].sum()
    # print("clicks:",clicks)

def plot_img(x_list, y_list, max_click_boundary):
    plt.plot(x_list, y_list, 'ro')
    plt.axis([0, 310, 0, max_click_boundary])
    plt.ylabel('clicks')
    plt.xlabel('constant bidding price')
    plt.savefig('constant_bidding.png')
    plt.show()

def constant_bidding(df, max_click_boundary):
    click_list = []

    for i in range(minBid+1, maxBid + 1):
        clicks = EvalBidClicksOnly(df, i, budget,size)
        click_list.append(clicks)
        print("bidding_price: {}; clicks: {}".format(i, clicks))

    df = pd.DataFrame({'bidding_price': list(range(1, maxBid + 1)), 'clicks': click_list})
    df.to_csv('constant_bidding_clicks_training_set.csv', encoding='utf-8', index=False)
    # df.to_csv('constant_bidding_clicks_validation_set.csv', encoding='utf-8', index=False)

    plot_img(list(range(1, maxBid + 1)), click_list, max_click_boundary)

def evaluation(constant_price, df):
    clicks = EvalBidClicksOnly(df, constant_price, budget, size)
    print(clicks)

constant_bidding(train_df, 700)
# constant_bidding(validation_df)
# evaluation(25, validation_df)
