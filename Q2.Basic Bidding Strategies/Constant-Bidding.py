import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


def constantBidding(dataset, constant_price):
    dataset['constantprice']=constant_price
    trueValues = (dataset["payprice"]<dataset["constantprice"])
    impressions = dataset[trueValues].shape[0]
    clicks = np.sum(dataset[trueValues]["click"].values)
    spend = np.sum(dataset[trueValues]["payprice"].values)

    CPC = spend/clicks
    CPM = spend/impressions
    CTR = (clicks*100)/impressions

    # print(constant_price, clicks, CTR, spend, CPM, CPC)
    resultList.append([constant_price, clicks, CTR, spend, CPM, CPC])
    # resultsDF.append([{'price': constant_price, 'clicks': clicks, 'CTR': CTR, 'spend': spend, 'CPM': CPM, 'CPC': CPC}], ignore_index = True)

def drawResult(dataset, column):
    plt.figure(figsize = (12,6))
    sns.pointplot(x = "price", y = column, data = dataset, color = "black", capsize = 0.2)
    plt.ylabel("")
    plt.title(column)
    plt.savefig("CB_"+column+".png")


pd.set_option('display.max_columns', None)
trainDF = pd.read_csv("../we_data/train.csv")
trainDF['constantprice'] = 0

minBid=np.min(trainDF["payprice"].values)
maxBid=np.max(trainDF["payprice"].values)
bidRange= np.arange(minBid, maxBid,10)

resultList = []
columnsList = ['price','clicks', 'CTR', 'spend', 'CPM', 'CPC']

for bidprice in bidRange:
    if bidprice > 0:
        constantBidding(trainDF, bidprice)

resultsDF = pd.DataFrame(resultList, columns=columnsList)

for column in columnsList:
    drawResult(resultsDF, column)

print(resultsDF)
