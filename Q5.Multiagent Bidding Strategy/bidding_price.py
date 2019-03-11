import pandas as pd
import numpy as np

test_df = pd.read_csv("../we_data/test.csv")

constant_df = pd.read_csv("./constant_bidding.csv")
random_df = pd.read_csv("./random_bidding.csv")


priceDF = pd.DataFrame(columns=['constant','random','linear','ortb'])
lists = {'constant':[], 'random':[], 'linear':[], 'ortb':[]}

constant = constant_df.bidding_price[constant_df['clicks'].idxmax()]
lowerBound = random_df.minValue[random_df['clicks'].idxmax()]
upperBound = random_df.maxValue[random_df['clicks'].idxmax()]

for i in range(0,len(test_df)):
        lists['constant'].append(constant)
        lists['random'].append(np.random.randint(lowerBound, upperBound+1))

priceDF.constant = lists['constant']
priceDF.random = lists['random']

priceDF.to_csv('price.csv', encoding='utf-8', index=False)
