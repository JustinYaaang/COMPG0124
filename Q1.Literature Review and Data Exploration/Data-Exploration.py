import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns



def aggregated_CTR(dataset, columns = "weekday"):
    feat = dataset[by].unique()
    res = pd.DataFrame(columns = [columns, "CTR", "Cost", "CPC"])
    i = 0

    for f in feat:
        dataset_ = dataset.loc[dataset[by] == f]

        impr = dataset_.shape[0]
        clicks = dataset_["click"].sum()
        ctr = clicks*100/impr
        cost = dataset_["payprice"].mean()

        if clicks > 0:
            cpc = dataset_["payprice"].sum() / clicks /  1000
        else:
            cpc = 0

        res.loc[i] = [f, ctr, cost, cpc]
        i+=1
    return res


def aggregated_statistics(dataset, columns):
    for column in columns:
        data = aggregated_CTR(dataset, column)
        data[column] = data[requirement].map(lambda x: int(x))
        data.sort_values(by = column)

        plt.figure(figsize = (6,6))
        sns.pointplot(x = column, y = "click", data = dataset, color = "black", capsize = 0.2)
        plt.ylabel("")
        plt.title("CTR")
        plt.savefig(column+"_CTR"+".png")

        plt.figure(figsize = (4,4))
        sns.pointplot(x = column, y = "payprice", data = dataset, color = "black", capsize = 0.2)
        plt.ylabel("")
        plt.title("Average Cost")
        plt.savefig(column+"_AC"+".png")

        plt.figure(figsize = (4,4))
        sns.pointplot(x = requirement, y = "CPC", data = data, color = "black", capsize = 0.2)
        plt.ylabel("")
        plt.title("CPC")
        plt.savefig(column+"_CPC"+".png")


pd.set_option('display.max_columns', None)
train = pd.read_csv("../we_data/train.csv")
# validation = pd.read_csv("../we_data/validation.csv")
# test = pd.read_csv("../we_data/test.csv")

columns=["weekday"]

aggregated_statistics(train, columns)
