import pandas as pd
import numpy as np
import os
import operator
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("../we_data/train.csv")
validation = pd.read_csv("../we_data/validation.csv")
test = pd.read_csv('../we_data/test.csv')

train['size'] = train['slotwidth'] * train['slotheight']
validation['size'] = validation['slotwidth'] * validation['slotheight']
test['size'] = test['slotwidth'] * test['slotheight']
train['OS'], train['browser'] = zip(*train['useragent'].map(lambda x: x.split('_')))
validation['OS'], validation['browser'] = zip(*validation['useragent'].map(lambda x: x.split('_')))
test['OS'], test['browser'] = zip(*test['useragent'].map(lambda x: x.split('_')))

def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))
    
    df.ix[df.slotprice.between(0, 10), 'slotpricebucket'] = 1
    df.ix[df.slotprice.between(11, 50), 'slotpricebucket'] = 2
    df.ix[df.slotprice.between(51, 100), 'slotpricebucket'] = 3
    df.ix[df.slotprice.between(101, 5000), 'slotpricebucket'] = 4
    df['slotpricebucket'] = df['slotpricebucket'].astype(np.uint8)

    pred=df.drop(['click','bidid','userid','IP','url','urlid','slotid','useragent','slotprice',
                 'bidprice','payprice','domain','slotwidth', 'slotheight'],axis=1)
    
    # create dummy variables for categoricals
    pred=pd.get_dummies(pred,dummy_na=True,columns=['weekday', 'hour', 
                                                       'OS', 'browser', 
                                                       'region', 'city', 'adexchange', 
                                                       'slotvisibility', 'slotformat',
                                                       'creative', 'slotpricebucket','advertiser'##'ip_block'
                                                    ,'keypage','size'])
    pred = pred.join(df.usertag.astype(str).str.strip('[]').str.get_dummies(','))
    pred=pred.drop(['usertag'],axis=1)
    print("After converting categoricals:\t{}".format(pred.shape))
    

    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(pred.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, pred.columns)
        print(to_add)
        print(to_drop)
        pred.drop(to_drop, axis=1, inplace=True)
        pred = pred.assign(**{c: 0 for c in to_add})
    
    pred.fillna(0, inplace=True)
    
    return pred


def pre_process_data_test(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))
    
    df.ix[df.slotprice.between(0, 10), 'slotpricebucket'] = 1
    df.ix[df.slotprice.between(11, 50), 'slotpricebucket'] = 2
    df.ix[df.slotprice.between(51, 100), 'slotpricebucket'] = 3
    df.ix[df.slotprice.between(101, 5000), 'slotpricebucket'] = 4
    df['slotpricebucket'] = df['slotpricebucket'].astype(np.uint8)

    pred=df.drop(['bidid','userid','IP','url','urlid','slotid','useragent','slotprice',
                 'domain','slotwidth', 'slotheight'],axis=1)
    
       # create dummy variables for categoricals
    pred = pd.get_dummies(pred,dummy_na=True,columns=['weekday', 'hour',  # ])
                                                       'OS', 'browser', 
                                                       'region', 'city', 'adexchange', 
                                                       'slotvisibility', 'slotformat',
                                                       'creative', 'slotpricebucket','advertiser'##'ip_block'
                                                    ,'keypage','size'])
    pred = pred.join(df.usertag.astype(str).str.strip('[]').str.get_dummies(','))
    pred = pred.drop(['usertag'],axis=1)
    print("After converting categoricals:\t{}".format(pred.shape))
    

    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(pred.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, pred.columns)
        print(to_add)
        print(to_drop)
        pred.drop(to_drop, axis=1, inplace=True)
        pred = pred.assign(**{c: 0 for c in to_add})
    
    pred.fillna(0, inplace=True)
    return pred

train_dum = pre_process_data(train)
valid_dum = pre_process_data(validation,enforce_cols=train_dum.columns)
test_dum = pre_process_data_test(test,enforce_cols=train_dum.columns)

y = train.click
y_validation = validation.click

def order(df_test, df_train):
    new_df = pd.DataFrame()
    
    for key in df_train:
        new_df[key] = df_test[key]
    return new_df
X_validation = order(valid_dum,train_dum)
X_test = order(test_dum,train_dum)

rus = RandomUnderSampler(random_state=3,ratio={1:1786,0:10716})
X_train,y_train = rus.fit_sample(train_dum,y)

model_LR = LogisticRegression(penalty = 'l1', max_iter = 100, C = 0.1,
                              solver = 'saga',class_weight = 'unbalanced')
model_LR.fit(X_train, y_train)

#X_validation = X_validation.as_matrix()
y_valid_pre = model_LR.predict_proba(X_validation)
pCTRval = pd.DataFrame(y_valid_pre)

#X_test = X_test.as_matrix()
y_test_pre = model_LR.predict_proba(X_test)
pCTRtest = pd.DataFrame(y_test_pre)

w=10716/train.shape[0]
avgCTR=sum(train.click)/train.shape[0]

test_score = y_test_pre[:,1]/(y_test_pre[:,1]+(1-y_test_pre[:,1])/w)
valid_score = y_valid_pre[:,1]/(y_valid_pre[:,1]+(1-y_valid_pre[:,1])/w)

# val_bid_ids = validation['bidid']
# test_bid_ids = test['bidid']

# valid_result = pd.concat([val_bid_ids, valid_score], axis=1)
# test_result = pd.concat([test_bid_ids, test_score], axis=1)

df = pd.DataFrame({'pCTR': valid_score})
df.to_csv('pCTRval_v3.csv', encoding='utf-8', index=False)

df = pd.DataFrame({'pCTR': test_score})
df.to_csv('pCTRtest_v3.csv', encoding='utf-8', index=False)

# test_result.to_csv('pCTRtest_v3.csv')
# valid_result.to_csv('pCTRval_v3.csv')

# eval_linear = pd.DataFrame(columns=['bid_base','Imps','spend','clicks'])

# max_num = 0
# max_bid = 0
# spend = 0
# iteration = 0

# for bid_base in np.arange(3,300, 3):
#     num_click = 0
#     flag = True
#     Imps = 0
#     spend = 0
#     iteration += 1
#     for i in range(validation.shape[0]):
#         bid = bid_base*(valid_score[i]/avgCTR)
#         if bid >= validation.payprice[i] and flag:
#             spend = spend + validation.payprice[i]
#             if spend > 6250000:
#                 spend = spend - validation.payprice[i]
#                 flag = False
#                 break
#             num_click = num_click + validation.click[i]
#             Imps = Imps + 1
#     eval_linear.loc[iteration,'bid_base'] = bid_base
#     eval_linear.loc[iteration,'clicks'] = num_click
#     eval_linear.loc[iteration,'spend'] = spend/1000
#     eval_linear.loc[iteration,'Imps'] = Imps

#     if num_click > max_num:
#         max_num = num_click
#         max_bid = bid_base
        
# eval_linear['CTR'] = eval_linear['clicks']/eval_linear['Imps']
# eval_linear['eCPC'] = eval_linear['spend']/eval_linear['clicks']
# eval_linear['CPM'] = eval_linear['spend']*1000/eval_linear['Imps']

































