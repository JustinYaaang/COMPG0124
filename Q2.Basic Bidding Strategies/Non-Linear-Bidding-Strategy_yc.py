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
    df_train = pd.read_csv('../we_data/train.csv')

    max_clicks = 0
    # size = df_train.shape[0]
    size = df_val.shape[0]

    click_list = []
    # c = 6
    # lamda = 5e-07
    lamda_list = [1e-10]
    c_write_list, lambda_write_list = [], [] 
    for i in range(100):
        lamda_list.append(lamda_list[-1] * 2)
    for c in range(1, 100):
        print(c)
        for lamda in lamda_list:
            bids = np.sqrt((c/lamda * pCTRval['pCTR']) + c**2) - c
            clicks = EvalBidByClicksOnly(df_val, bids, budget, size)
            if clicks > max_clicks:
                max_clicks = clicks
                best_c = c
                best_lamda = lamda
            click_list.append(clicks)
            c_write_list.append(c)
            lambda_write_list.append(lamda)


    print("best clicks: {}, c: {}, lamda: {}".format(max_clicks, best_c, best_lamda))
    df = pd.DataFrame({'c': c_write_list, 'lamda': lambda_write_list, 'clicks': click_list})
    df.to_csv('non_linear_bidding.csv', encoding='utf-8', index=False)

    

def non_linear_bidding_v2():
    pCTRval = pd.read_csv('pCTRval_retrieved.csv')
    df_val = pd.read_csv('../we_data/validation.csv')
    df_train = pd.read_csv('../we_data/train.csv')

    max_clicks = 0
    # size = df_train.shape[0]
    size = df_val.shape[0]

    click_list = []
    # c = 6
    # lamda = 5e-07
    lamda_list = [1e-10]
    c_write_list, lambda_write_list = [], [] 
    for i in range(40):
        lamda_list.append(lamda_list[-1] * 1.5)
    # lamda_list = [1.20e-05, 1.21e-05, 1.22e-05, 1.23e-05, 1.24e-05, 1.25e-05, 1.26e-05, 1.27e-05, 1.28e-05, 1.29e-05, 1.3e-05, 1.31e-05]
    # for c in range(10, 200, 10):
    lamda_list = [3.6e-06, 3.7e-06, 3.8e-06, 3.9e-06]
    for c in range(120, 130, 1):
        print(c)
        for lamda in lamda_list:
            bids = np.sqrt((c/lamda * pCTRval['pCTR']) + c**2) - c
            clicks = EvalBidByClicksOnly(df_val, bids, budget, size)
            if clicks > max_clicks:
                max_clicks = clicks
                best_c = c
                best_lamda = lamda
            click_list.append(clicks)
            c_write_list.append(c)
            lambda_write_list.append(lamda)

    print("best clicks: {}, c: {}, lamda: {}".format(max_clicks, best_c, best_lamda))
    df = pd.DataFrame({'c': c_write_list, 'lamda': lambda_write_list, 'clicks': click_list})
    df.to_csv('non_linear_bidding_v2.csv', encoding='utf-8', index=False)

def non_linear_on_test():
    #pCTRval = pd.read_csv('pCTRtest2.csv', delimiter=';', sep='\s*,\s*', encoding="utf-8-sig")
    pCTRval = pd.read_csv('pCTRtest.csv')
    df_val = pd.read_csv('../we_data/validation.csv')
    size = df_val.shape[0]

    print("COOL")
    
    #df_val = pd.read_csv('../we_data/validation.csv')

    #avgCTR = df_val.click.sum() / df_val.bidid.count()
    #print(pCTRval.pCTR.sum() / df_val.bidid.count())
    #print(avgCTR)
    #size = df_val.shape[0]

    c = 18
    lamda = 1.6384e-06

    bids = np.sqrt((c/lamda * pCTRval['pCTR']) + c**2) - c 
    # bid_ids = pCTRval['bidid']

    # res = pd.concat([bid_ids, bids], axis=1, sort=False)
    # print(bids.sum())
    # print(bids[:10])
    # res.to_csv('submission1.csv')



    clicks = EvalBidByClicksOnly(df_val, bids, budget, size)
    print("clicks: {}, c: {}, lamda: {}".format(clicks, c, lamda))

non_linear_on_test()

# non_linear_bidding()
# non_linear_bidding_v2()

# best clicks: 39, c: 83, lamda: 1.31072e-05

# best clicks: 39, c: 82, lamda: 1.27e-05

# best clicks: 119, c: 127, lamda: 4e-06

# best clicks: 124, c: 126, lamda: 3.8e-06




