import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
budget=6250*1000


def constant_bidding_price_generation(m):
    price_list = []
    for i in range(1, m+1):
        price_list.append(100+10*i)
    # print(price_list)
    return price_list


def random_bidding_price_generation(m, diff):
    price_list = []
    for i in range(1, m+1):
        price_list.append([i*10, i*10+diff])
    return price_list


def linear_bidding_price_generation(m):
    price_list = []
    for i in range(1, m+1):
        price_list.append(10*i)
    return price_list


def get_random_bids(lower, upper, nb_rows, nb_agents):
    return np.random.randint(lower, upper, (nb_rows, nb_agents))


def generate_m_constant_bidding_agents(m, AdjustedBudget, dataframe):
    ## m could be 20
    budget_list = [AdjustedBudget] * m
    price_list = constant_bidding_price_generation(m)
    all_bids = np.full((dataframe.shape[0], len(price_list)), price_list)
    return budget_list, all_bids


def generate_m_random_bidding_agents(m, AdjustedBudget, diff, dataframe, lower, upper):
    ## m = 20
    budget_list = [AdjustedBudget] * m
    all_bids = get_random_bids(lower, upper+1, dataframe.shape[0], m)
    return budget_list, all_bids


def generate_m_linear_bidding_agents(m, AdjustedBudget, avgCTR, df_pCTR):
    ## m could be 20
    budget_list = [AdjustedBudget] * m
    price_list = linear_bidding_price_generation(m)
    bids = []

    for i in range(m):
        bids.append(df_pCTR['pCTR'] * price_list[i] / avgCTR)
    bids = np.array(bids).T

    return budget, bids


def generate_m_non_linear_bidding_agents(m, AdjustedBudget, pCTRval):
    ## m could be 20
    budget_list = [AdjustedBudget] * m
    c = 98
    lamda = 1.28e-08

    bids = np.sqrt((c/lamda * pCTRval['pCTR']) + c**2) - c 
    all_bids = []
    for i in range(m):
        all_bids.append(bids)

    all_bids = np.array(all_bids).T
    # print(all_bids.shape)
    return budget_list, all_bids


def multi_agent_simulation(all_bids, all_budgets, df_val, n):
    payprices = df_val['payprice']
    click = df_val['click']
    columns_of_maxes = np.argmax(all_bids, axis=1) 
    click_list = [0] * n

    for i in range(len(columns_of_maxes)):
        index_of_max = columns_of_maxes[i]
        if all_budgets[index_of_max] >= all_bids[i,index_of_max] >= payprices[i]:
            click_list[index_of_max] += click[i]
            # if 19 < index_of_max <= 39:
            #     print(i)
            all_budgets[index_of_max] -= all_bids[i,index_of_max]
        elif all_bids[i,index_of_max] >= payprices[i] > all_budgets[index_of_max]:
            all_bids[:, index_of_max] = 0
            columns_of_maxes = np.argmax(all_bids, axis=1) 

    print(click_list)


def main():
    df_train = pd.read_csv('../we_data/train.csv')
    df_val = pd.read_csv('../we_data/validation.csv')
    df_test = pd.read_csv('../we_data/test.csv')
    df_pCTR = pd.read_csv('pCTRval_v3.csv')

    print("data read")

    dataframe = df_val
    size = df_val.shape[0]
    AdjustedBudget=(budget/size)*dataframe.shape[0]
    avgCTR = df_train.click.sum() / df_train.bidid.count()
    lower, upper = 170, 300

    m = 20
    n = 80

    nonlinear_budgets, nonlinear_bids = generate_m_non_linear_bidding_agents(m, AdjustedBudget, df_pCTR)
    print("non-linear bidding generation DONE")

    random_budgets, random_bids = generate_m_random_bidding_agents(m, AdjustedBudget, 100, dataframe, lower, upper)
    print("random bidding generation DONE")
    
    constant_budgets, constant_bids = generate_m_constant_bidding_agents(m, AdjustedBudget, dataframe)
    print("constant bidding generation DONE")

    linear_budgets, linear_bids = generate_m_linear_bidding_agents(m, AdjustedBudget, avgCTR, df_pCTR)
    print("linear bidding generation DONE")


    print(random_bids.shape)
    print(constant_bids.shape)
    print(linear_bids.shape)
    print(nonlinear_bids.shape)

    all_bids = np.concatenate((random_bids, constant_bids, linear_bids, nonlinear_bids), axis=1)
    # all_budgets = random_budgets + constant_budgets + linear_budgets + nonlinear_budgets
    all_budgets = [budget] * n

    print(all_bids.shape)
    print(len(all_budgets))

    print("start simulation")

    multi_agent_simulation(all_bids, all_budgets, df_val, n)


# main()

a = [4, 1, 3, 2, 0, 1, 6, 5, 2, 2, 0, 4, 1, 3, 5, 4, 4, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 6, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 28, 26, 24, 14, 6, 1, 1, 3, 1, 2, 0, 2, 3, 1, 1, 0, 5, 6, 3, 5, 0, 3, 1, 2]
for i in range(80):
    if a[i] > 0:
        print("i: {}, clicks: {}".format(i, a[i]))






    