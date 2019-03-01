import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import math
import random

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, \
                            precision_score, make_scorer
from sklearn.externals import joblib




def UsertagCategories(df):
    
    # Drop nan
    df = df["usertag"].dropna().reset_index(drop = True)
    
    # Find unique usertags
    usertags_list = [df[i].split(",") for i in range(df.shape[0])]
    
    # itertools.chain.from_iterable joins a list of lists into a single list
    usertags = np.unique(list(itertools.chain.from_iterable(usertags_list)))
    
    # Remove the empty string ""
    usertags = [tag for tag in usertags if len(tag) > 0]
    
    return usertags


def FeatureEngineering(df):
    
        # Convert numerical to categorical
        df["weekday_cat"] = df["weekday"].map(lambda x: str(x))
        df["hour_cat"] = df["hour"].map(lambda x: str(x))
        df["region_cat"] = df["region"].map(lambda x: str(x))
        df["city_cat"] = df["city"].map(lambda x: str(x))
        df["adexchange_cat"] = df["adexchange"].map(lambda x: str(x))
        df["advertiser_cat"] = df["advertiser"].map(lambda x: str(x))
        
        # Operating system
        df["os"] = df["useragent"].map(lambda x: x.split("_")[0])
        
        # Browser
        df["browser"] = df["useragent"].map(lambda x: x.split("_")[1])
        
        # Slotarea
        df["slotarea"] = df["slotwidth"]*df["slotheight"]   ####.astype("category")
        
        # Part of the day
        df["part_of_the_day"] = ""
        
        df.loc[(df["weekday"] == 0) & (df["hour"] > 0) & (df["hour"] <=8), "part_of_the_day"] = "Sunday_Night"
        df.loc[(df["weekday"] == 0) & (df["hour"] > 8) & (df["hour"] < 17), "part_of_the_day"] = "Sunday_Morning"
        df.loc[(df["weekday"] == 0) & (((df["hour"] >= 17) & (df["hour"] <= 23)) | ((df["hour"] == 0))), "part_of_the_day"] = "Sunday_Evening"
        
        df.loc[(df["weekday"] == 1) & (df["hour"] > 0) & (df["hour"] <=8), "part_of_the_day"] = "Monday_Night"
        df.loc[(df["weekday"] == 1) & (df["hour"] > 8) & (df["hour"] < 17), "part_of_the_day"] = "Monday_Morning"
        df.loc[(df["weekday"] == 1) & (((df["hour"] >= 17) & (df["hour"] <= 23)) | ((df["hour"] == 0))), "part_of_the_day"] = "Monday_Evening"

        
        df.loc[(df["weekday"] == 2) & (df["hour"] > 0) & (df["hour"] <=8), "part_of_the_day"] = "Tuesday_Night"
        df.loc[(df["weekday"] == 2) & (df["hour"] > 8) & (df["hour"] < 17), "part_of_the_day"] = "Tuesday_Morning"
        df.loc[(df["weekday"] == 2) & (((df["hour"] >= 17) & (df["hour"] <= 23)) | ((df["hour"] == 0))), "part_of_the_day"] = "Tuesday_Evening"

        df.loc[(df["weekday"] == 3) & (df["hour"] > 0) & (df["hour"] <=8), "part_of_the_day"] = "Wednesday_Night"
        df.loc[(df["weekday"] == 3) & (df["hour"] > 8) & (df["hour"] < 17), "part_of_the_day"] = "Wednesday_Morning"
        df.loc[(df["weekday"] == 3) & (((df["hour"] >= 17) & (df["hour"] <= 23)) | ((df["hour"] == 0))), "part_of_the_day"] = "Wednesday_Evening"

        df.loc[(df["weekday"] == 4) & (df["hour"] > 0) & (df["hour"] <=8), "part_of_the_day"] = "Tuesday_Night"
        df.loc[(df["weekday"] == 4) & (df["hour"] > 8) & (df["hour"] < 17), "part_of_the_day"] = "Tuesday_Morning"
        df.loc[(df["weekday"] == 4) & (((df["hour"] >= 17) & (df["hour"] <= 23)) | ((df["hour"] == 0))), "part_of_the_day"] = "Tuesday_Evening"

        df.loc[(df["weekday"] == 5) & (df["hour"] > 0) & (df["hour"] <=8), "part_of_the_day"] = "Friday_Night"
        df.loc[(df["weekday"] == 5) & (df["hour"] > 8) & (df["hour"] < 17), "part_of_the_day"] = "Friday_Morning"
        df.loc[(df["weekday"] == 5) & (((df["hour"] >= 17) & (df["hour"] <= 23)) | ((df["hour"] == 0))), "part_of_the_day"] = "Friday_Evening"

        
        df.loc[(df["weekday"] == 6) & (df["hour"] > 0) & (df["hour"] <=8), "part_of_the_day"] = "Saturday_Night"
        df.loc[(df["weekday"] == 6) & (df["hour"] > 8) & (df["hour"] < 17), "part_of_the_day"] = "Saturday_Morning"
        df.loc[(df["weekday"] == 6) & (((df["hour"] >= 17) & (df["hour"] <= 23)) | ((df["hour"] == 0))), "part_of_the_day"] = "Saturday_Evening"
        
        # Slotprice binning
        df["slotprice_cat"] = 0
        
        df.loc[ df["slotprice"] <= 10, "slotprice_cat"] = 0
        df.loc[ (df["slotprice"] > 10) & (df["slotprice"] <= 50), "slotprice_cat"] = 1
        df.loc[ (df["slotprice"] > 50) & (df["slotprice"] <= 100), "slotprice_cat"] = 2
        df.loc[ df["slotprice"] > 100, "slotprice_cat"] = 3
        
        return df


def DropColumns(df):
    
    columns = ["weekday", "hour", "bidid", "userid", "useragent", "IP", "domain", "url", "urlid", "slotid",
               "slotwidth", "slotheight", "slotprice", "keypage", "usertag", "region", "city", "adexchange", "advertiser"]
    df.drop(columns, axis = 1, inplace = True)
    
    return df


def GetDummies(df):
    
    df = pd.get_dummies(df)
    return df


if __name__ == '__main__':
    name_helper = "pCTR___"
    train = pd.read_csv("../we_data/train.csv")
    # train = pd.read_csv("../we_data/validation.csv")
    validation = pd.read_csv("../we_data/validation.csv")
    all_data = pd.concat((train, validation), axis = 0)
    all_data = FeatureEngineering(all_data)
    all_data = DropColumns(all_data)
    all_data_dummy = GetDummies(all_data)
    train_dummy = all_data_dummy[:train_2.shape[0]]
    validation_dummy = all_data_dummy[train_2.shape[0]:]
    


    df.to_csv('feature_extraction.csv', encoding='utf-8', index=False)
    # print(df)


