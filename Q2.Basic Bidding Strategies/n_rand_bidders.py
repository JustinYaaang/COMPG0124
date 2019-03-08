import agent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint 
from agent import Agent

budget=6250*1000

def get_random_bids(lower, upper, nb_rows, nb_agents):
    return np.random.randint(lower, upper, (nb_rows, nb_agents))

def get_payprices(dataframe):
    return dataframe['payprice']

def get_clicks(dataframe):
    return dataframe['click']

def get_agents(nb_agents):
    agents = list()
    for i in range(0, nb_agents): 
        ag = Agent(i)
        agents.append(ag)
    return agents




def simulate_n_random_bidders(lower, upper, dataframe, nb_agents):
    # dataframe.shape[0]: nb of rows, i.e. nb of examples
    all_bids = get_random_bids(lower, upper, dataframe.shape[0], nb_agents) 

    agents = get_agents(nb_agents)

    payprice = get_payprices(dataframe)
    all_bids = np.column_stack((all_bids, payprice))

    clicks = get_clicks(dataframe)
    print(np.sum(clicks))
    # This contains, for each row, what the index of the max is.
    columns_of_maxes = np.argmax(all_bids, axis=1) 

    for i, index_of_max in enumerate(columns_of_maxes):
        if index_of_max == nb_agents: # then the max is the bid price, so no agent win this bid
            continue
        click_won = clicks[i]
        agents[index_of_max].addClicks(click_won)

    clicks_per_agent = list()
    for ag in agents:
        clicks_per_agent.append(ag.clicks)

    return clicks_per_agent



def simulate_n_random_bidders_v2(lower, upper, dataframe, nb_agents, size):
    # dataframe.shape[0]: nb of rows, i.e. nb of examples
    all_bids = get_random_bids(lower, upper+1, dataframe.shape[0], nb_agents) 
    agents = get_agents(nb_agents)
    payprice = get_payprices(dataframe)
    AdjustedBudget=(budget/size)*dataframe.shape[0]

    used_budgets, total_clicks = [0] * nb_agents, 0


    clicks = get_clicks(dataframe)
    # print(np.sum(clicks))
    # This contains, for each row, what the index of the max is.
    columns_of_maxes = np.argmax(all_bids, axis=1) 

    for i, index_of_max in enumerate(columns_of_maxes):
        if all_bids[i,index_of_max] >= payprice[i]:
            used_budgets[index_of_max] += all_bids[i, index_of_max]
            total_clicks += clicks[i]
        # else:
        #     print("payprice[i]: {}, all_bids[i, index_of_max]: {}".format(payprice[i], all_bids[i, index_of_max]))
    # print(used_budgets)
    # print("lower bound: {}, upper bound: {}, the max budget spent: {}".format(lower, upper, max(used_budgets)))
    # print(total_clicks)
    return total_clicks if max(used_budgets) < budget else 0


def main():
    print("main")
    # these bounds were found from the previous section
    lower = 30
    upper = 110
    nb_agents = 100

    dataframe = pd.read_csv('../we_data/validation.csv')

    clicks_per_agent = simulate_n_random_bidders(lower, upper, dataframe, nb_agents)
    print(clicks_per_agent)
    count_list = [0] * 20
    for i in range(nb_agents):
        # print(clicks_per_agent[i])
        count_list[clicks_per_agent[i]] += 1
    print(count_list)
    print(sum(count_list))
    plt.plot(count_list)
    plt.show()
    print(sum(clicks_per_agent))

def main2(nb_agents):
    print("agent number: {}".format(nb_agents))
    # lower = 30
    # upper = 110
    dataframe = pd.read_csv('../we_data/validation.csv')
    df_train = pd.read_csv('../we_data/train.csv')
    size = df_train.shape[0]
    max_clicks = 0

    # print("max")
    # print(np.amax(get_payprices(dataframe)))

    lower_list, upper_list, click_list = [], [], []
    best_lower, best_upper = 0, 0
    for lower in range(50, 300, 20):
        print("lower: {}".format(lower))
        for add in range(10, 260, 20):
            upper = lower + add
            if upper > 300:
                continue

            cur_clicks = simulate_n_random_bidders_v2(lower, upper, dataframe, nb_agents, size)
            if max_clicks < cur_clicks:
                max_clicks = cur_clicks
                best_lower = lower
                best_upper = upper
            lower_list.append(lower)
            upper_list.append(upper)
            click_list.append(cur_clicks)
    
    df = pd.DataFrame({'upper': upper_list, 'lower': lower_list, 'clicks': click_list})
    df.to_csv('n_random_bidder2.csv', encoding='utf-8', index=False)

    print("max_clicks: {}, lower: {}, upper: {}".format(max_clicks, best_lower, best_upper))


if __name__ == "__main__":
    for i in range(60, 110, 10):
        main2(i)

# 50
# max_clicks: 202, lower: 70, upper: 300
# 60
# max_clicks: 202, lower: 50, upper: 300
# 70
# max_clicks: 202, lower: 90, upper: 300
# 80
# max_clicks: 202, lower: 170, upper: 300
# 90
# max_clicks: 202, lower: 90, upper: 300
# 100
# max_clicks: 202, lower: 70, upper: 300








