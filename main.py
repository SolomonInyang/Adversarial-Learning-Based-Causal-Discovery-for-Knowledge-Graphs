from causal_discovery import Q_Mat
import networkx as nx
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from SAM_implementation import SAM


def generate_matrix():
    q_obj = Q_Mat("fb15k-237", "-", "True")
    q_mats = q_obj.initialize()

    adj_matrix = np.array(q_mats[0])
    df = pd.DataFrame(adj_matrix)

    # rename column names
    label_word = 'node_'

    df = df.rename(columns=lambda x: label_word + str(x))

    # number of nodes to select
    num_nodes = 10

    # get first row (1 relation type) and 5 nodes
    t_df = df.iloc[:, :num_nodes]

    return t_df


train_df = generate_matrix()

# predict based on data
obj = SAM()
output = obj.predict(train_df)

# View the generated causal graph
nx.draw_networkx(output, font_size=8)
plt.show()
