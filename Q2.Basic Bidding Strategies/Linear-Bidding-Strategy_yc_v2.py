import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression 

df_train = pd.read_csv("../we_data/train.csv")
df_test = pd.read_csv('../we_data/test.csv')
df_val = pd.read_csv('../we_data/validation.csv')

print("data read")

def encode_days(dataframe):
    dataframe = pd.concat([dataframe,pd.get_dummies(dataframe.weekday,prefix='day')],axis=1)
    dataframe = dataframe.drop('weekday',axis=1)
    return dataframe

def encode_hours(dataframe):
    dataframe = pd.concat([dataframe,pd.get_dummies(dataframe.hour,prefix='hour')],axis=1)
    dataframe = dataframe.drop('hour',axis=1)
    return dataframe

def encode_region(dataframe):
    dataframe = pd.concat([dataframe,pd.get_dummies(dataframe.region,prefix='region')],axis=1)
    dataframe = dataframe.drop('region',axis=1)
    return dataframe

def encode_adexchange(dataframe):
    dataframe = pd.concat([dataframe,pd.get_dummies(dataframe.adexchange,prefix='adexchange')],axis=1)
    dataframe = dataframe.drop('adexchange',axis=1)
    return dataframe

def encode_slotwidth(dataframe):
    dataframe = pd.concat([dataframe,pd.get_dummies(dataframe.slotwidth,prefix='slotwidth')],axis=1)
    dataframe = dataframe.drop('slotwidth',axis=1)
    return dataframe

def encode_slotheight(dataframe):
    dataframe = pd.concat([dataframe,pd.get_dummies(dataframe.slotheight,prefix='slotheight')],axis=1)
    dataframe = dataframe.drop('slotheight',axis=1)
    return dataframe

def encode_advertiser(dataframe):
    dataframe = pd.concat([dataframe,pd.get_dummies(dataframe.advertiser,prefix='advertiser')],axis=1)
    dataframe = dataframe.drop('advertiser',axis=1)
    return dataframe

def encode_slotvisibility(dataframe):
    dataframe = pd.concat([dataframe,pd.get_dummies(dataframe.slotvisibility,prefix='slotvisibility')],axis=1)
    dataframe = dataframe.drop('slotvisibility',axis=1)
    return dataframe

def encode_slotformat(dataframe):
    dataframe = pd.concat([dataframe,pd.get_dummies(dataframe.slotformat,prefix='slotformat')],axis=1)
    dataframe = dataframe.drop('slotformat',axis=1)
    return dataframe

def encode_os_browser(dataframe):
    df_temp = pd.DataFrame(dataframe.useragent.str.split('_',1).tolist(), columns = ['OS','browser'])
    dataframe = pd.concat([dataframe,df_temp],axis=1)
    dataframe = dataframe.drop('useragent',axis=1)
    dataframe = pd.concat([dataframe,pd.get_dummies(dataframe.OS,prefix='OS')],axis=1)
    dataframe = dataframe.drop('OS',axis=1)
    dataframe = pd.concat([dataframe,pd.get_dummies(dataframe.browser,prefix='browser')],axis=1)
    dataframe = dataframe.drop('browser',axis=1)
    return dataframe

# 12. Encode slotprice into 5 ranges
def encode_slotprice(dataframe):
    slotprice_range = pd.DataFrame()
    slotprice_range['slotprices'] = pd.cut(dataframe.slotprice.values,5, labels=[1,2,3,4,5])
    dataframe = pd.concat([dataframe,slotprice_range],axis=1)
    dataframe = pd.concat([dataframe,pd.get_dummies(dataframe.slotprices,prefix='slotprice')],axis=1)
    dataframe = dataframe.drop('slotprice',axis=1)
    dataframe = dataframe.drop('slotprices',axis=1)
    return dataframe

def encode_usertags(dataframe):
    usertags = list(dataframe.usertag)
    unique_users = set()
    list_users = []
    for user in usertags:
        u = user.split(',')
        list_users.append(u)
        for us in u:
            unique_users.add(us)
    users = pd.DataFrame()
    for user in unique_users:
        users["user_"+user] = 0
    dataframe = pd.concat([dataframe,users],axis=1)
    for user in unique_users:
        datas = []
        for users in list_users:
            if user in users:
                datas.append(1)
            else:
                datas.append(0)
        dataframe["user_"+user] = datas
    dataframe = dataframe.drop('usertag',axis=1)
    return dataframe


xtrain = df_train.drop(['click','bidid','userid','IP','city','domain', 'url','urlid','slotid','creative','bidprice','payprice','keypage', 'usertag'], axis=1)
ytrain = df_train.click

xval = df_val.drop(['click','bidid','userid','IP','city','domain', 'url','urlid','slotid','creative','bidprice','payprice','keypage', 'usertag'], axis=1)
yval = df_val.click

xtest = df_test.drop(['bidid','userid','IP','city','domain', 'url','urlid','slotid','creative','keypage', 'usertag'], axis=1)

xtrain = encode_adexchange(xtrain)
xtrain = encode_advertiser(xtrain)
xtrain = encode_days(xtrain)
xtrain = encode_hours(xtrain)
xtrain = encode_os_browser(xtrain)
xtrain = encode_region(xtrain)
xtrain = encode_slotformat(xtrain)
xtrain = encode_slotheight(xtrain)
xtrain = encode_slotprice(xtrain)
xtrain = encode_slotvisibility(xtrain)
xtrain = encode_slotwidth(xtrain)
# xtrain = encode_usertags(xtrain)

xval = encode_adexchange(xval)
xval = encode_advertiser(xval)
xval = encode_days(xval)
xval = encode_hours(xval)
xval = encode_os_browser(xval)
xval = encode_region(xval)
xval = encode_slotformat(xval)
xval = encode_slotheight(xval)
xval = encode_slotprice(xval)
xval = encode_slotvisibility(xval)
xval = encode_slotwidth(xval)
# xval = encode_usertags(xval)

xtest = encode_adexchange(xtest)
xtest = encode_advertiser(xtest)
xtest = encode_days(xtest)
xtest = encode_hours(xtest)
xtest = encode_os_browser(xtest)
xtest = encode_region(xtest)
xtest = encode_slotformat(xtest)
xtest = encode_slotheight(xtest)
xtest = encode_slotprice(xtest)
xtest = encode_slotvisibility(xtest)
xtest = encode_slotwidth(xtest)
# xtest = encode_usertags(xtest)

#run on validation set
model = LogisticRegression(penalty='l2', class_weight='balanced')
resultval = model.fit(xtrain, ytrain).predict(xval)

#run on test set
resulttest = model.fit(xtrain, ytrain).predict(xtest)

predprob = model.predict_proba(xval)
pCTRval = pd.DataFrame(predprob)

#print AUc score
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(df_val.click, pCTRval[1])
metrics.auc(fpr, tpr)

predprob = model.predict_proba(xtest)

pCTRtest = pd.DataFrame(predprob)

#recalibrate the pctr
#newpctr = pctr / ( pctr + (1-pctr)/balance_ratio)

new_pctrval = []
new_pctrtest = []

ratio = len(df_train) / 2 * np.bincount(df_train.click)
balance_ratio = ratio[1] / ratio[0]

for pctr in pCTRval[1]:
    new_pctrval.append( pctr / (pctr + ((1-pctr) / balance_ratio)))

for pctr in pCTRtest[1]:
    new_pctrtest.append( pctr / (pctr + ((1-pctr) / balance_ratio)))

new_pctrval = pd.DataFrame(new_pctrval)
new_pctrval.to_csv('pCTRval.csv')

new_pctrtest = pd.DataFrame(new_pctrtest)
new_pctrtest.to_csv('pCTRtest.csv')

fpr, tpr, thresholds = metrics.roc_curve(df_val.click, new_pctrval)
metrics.auc(fpr, tpr)

f, axes = plt.subplots(1, figsize=(8, 5))
lab = 'AUC=%.5f' % metrics.auc(fpr, tpr)
axes.step(fpr, tpr, lw=2,label=lab)
axes.legend(loc='lower right', fontsize='small')
plt.show()



