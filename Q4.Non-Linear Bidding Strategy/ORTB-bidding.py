from dataReader import dataReader
from predictCTR import predictCTR
import pandas as pd
import numpy as np

def ortb_bid_generator():
    #formula = sqrt(c/lambda pctr + c**2) - c
    c = 77
    lamda = 3.3e-06
    bids = []

    impr = pClickDF.shape[0]
    clicks = pClickDF['click'].sum()
    ctr = clicks*100/impr

    for p in pCTRDF['pCTR']:
        bid = np.sqrt((c/lamda * p) + c**2) - c
        bids.append(bid)
    return bids

if __name__ == "__main__":
    validationset = "../we_data/validation.csv"
    pClickset = "./pClick.csv"
    pCTRset = "./pCTRval.csv"

    validationReader = dataReader(validationset)
    pClickReader = dataReader(pClickset)
    pCTRReader = dataReader(pCTRset)

    validationDF = validationReader.getDataFrame()
    pClickDF = pClickReader.getDataFrame()
    pCTRDF = pCTRReader.getDataFrame()

    ortbsDF = pd.DataFrame()
    ortbsDF['bids'] = ortb_bid_generator()
    newValidationDF = pd.concat([validationDF, ortbsDF],axis=1)
    resultDF = pd.DataFrame(columns=['clicks','imps','spent','CTR','CPC','CPM'])
    lists = {'clicks':[], 'imps':[], 'spent':[], 'ctr':[], 'cpc':[], 'cpm':[]}

    budget = 6250000.0
    imps = 0
    clicks = 0
    spent = 0
    for i in range(0,len(newValidationDF)):
        payprice = newValidationDF.payprice[i]
        bid = newValidationDF.bids[i]
        if budget < payprice:
            break
        if bid >= payprice:
            imps = imps + 1
            clicks = clicks + newValidationDF.click[i]
            budget = budget - payprice
            spent = spent + payprice
    spent = spent / 1000
    ctr = ((clicks / imps) * 100).round(4).astype(str)
    cpm = ((spent / imps) * 1000).round(4).astype(str)
    cpc = (spent / clicks).round(4).astype(str)
    lists['clicks'].append(clicks)
    lists['imps'].append(imps)
    lists['spent'].append(spent)
    lists['ctr'].append(ctr)
    lists['cpc'].append(cpc)
    lists['cpm'].append(cpm)

    resultDF.clicks = lists['clicks']
    resultDF.imps = lists['imps']
    resultDF.spent = lists['spent']
    resultDF.CTR = lists['ctr']
    resultDF.CPC = lists['cpc']
    resultDF.CPM = lists['cpm']
    print(resultDF)
