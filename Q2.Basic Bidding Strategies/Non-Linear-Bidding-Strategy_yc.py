import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
budget=6250*1000

def EvalBidByClicksOnly(dataframe, bids, budget, size):
    AdjustedBudget=(budget/size)*dataframe.shape[0]
    tempData=dataframe

    tempData['ConstBid'] = bids
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


def non_linear_bidding():
    pCTRval = pd.read_csv('pCTRval_retrieved.csv')
    df_val = pd.read_csv('../we_data/validation.csv')

    avgCTR = df_val.click.sum() / df_val.bidid.count()
    print(pCTRval.pCTR.sum() / df_val.bidid.count())
    print(avgCTR)
    lower_bound, upper_bound = 1, 300
    best_base_bid, max_clicks = 1, 0
    size = df_val.shape[0]
    click_list = []
    # c = 6
    # lamda = 5e-07
    # lamda_list = [1e-10]
    # for i in range(100):
    #     lamda_list.append(lamda_list[-1] * 2)
    # for c in range(18, 50):
    #     for lamda in lamda_list:
    #         bids = np.sqrt((c/lamda * pCTRval['pCTR']) + c**2) - c
    #         clicks = EvalBidByClicksOnly(df_val, bids, budget, size)
    #         if clicks > max_clicks:
    #             max_clicks = clicks
    #             best_c = c
    #             best_lamda = lamda
    #         click_list.append(clicks)

            #print("clicks: {}".format(clicks))
    c = 18
    lamda = 1.6384e-06
    bids = np.sqrt((c/lamda * pCTRval['pCTR']) + c**2) - c
    clicks = EvalBidByClicksOnly(df_val, bids, budget, size)
    print("clicks: {}, c: {}, lamda: {}".format(clicks, c, lamda))
    print("total clicks: {}".format(df_val.click.sum()))

    #print(click_list)
    #print("max_clicks: {}, c: {}, lamda: {}".format(max_clicks, best_c, best_lamda))

def non_linear_on_test():
    #pCTRval = pd.read_csv('pCTRtest2.csv', delimiter=';', sep='\s*,\s*', encoding="utf-8-sig")
    pCTRval = pd.read_csv('pCTRtest.csv')
    print("COOL")
    
    #df_val = pd.read_csv('../we_data/validation.csv')

    #avgCTR = df_val.click.sum() / df_val.bidid.count()
    #print(pCTRval.pCTR.sum() / df_val.bidid.count())
    #print(avgCTR)
    #size = df_val.shape[0]

    c = 18
    lamda = 1.6384e-06
    bids = np.sqrt((c/lamda * pCTRval['pCTR']) + c**2) - c 
    bid_ids = pCTRval['bidid']

    res = pd.concat([bid_ids, bids], axis=1, sort=False)
    print(bids.sum())
    print(bids[:10])
    #clicks = EvalBidByClicksOnly(df_val, bids, budget, size)
    #print("clicks: {}, c: {}, lamda: {}".format(clicks, c, lamda))
    #print("total clicks: {}".format(df_val.click.sum()))
    res.to_csv('submission1.csv')

non_linear_on_test()



