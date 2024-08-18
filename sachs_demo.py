import networkx as nx
from SAM_implementation import SAM
from data_loader import load_dataset
import matplotlib.pyplot as plt


data, graph = load_dataset("sachs")

# predict based on sachs data
obj = SAM()
output = obj.predict(data)

# View the generated causal graph
nx.draw_networkx(output, font_size=8)
plt.show()
