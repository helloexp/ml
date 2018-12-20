# coding=utf-8

import networkx as nx

import matplotlib.pyplot as  plt

G=nx.Graph()

G.add_edge(1,2)

nx.draw_networkx(G)

plt.show()
