#!/bin/python3
import sys
from graph_lib import Graph

graph = Graph()
try:
    graph = graph.load("graph.pickle")
except FileNotFoundError:
    N = 15
    #M = 30
    gamma = 1.2
    degree = 4
    levels = 3
    while True:
        #graph.init_powerlaw(N,gamma)
        #graph.init_random(N,M)
        #graph.init_regular(N,degree)
        graph.init_cayley_tree(degree,levels)
        break
        #graph.dijkstra_all(0)
        #if graph.is_connected:
            #break
        #print("! graph is not connnected. Initializing again !")
    #graph.set_gaussian_edges_x0
    graph.relax(crunchMode="exp")
    graph.save("graph.pickle")
graph.print()

start_node = 41
end_node = 32

#graph.set_constant_edges_lengths()
#graph.dijkstra_all(start_node)
#print(f"graph.dijkstra_dist from node {start_node:d} =",graph.dijkstra_dist)
#print(f"graph.dijkstra_prev from node {start_node:d} =",graph.dijkstra_prev)

graph.plot_dijkstra(start_node, end_node)

#graph.set_distance_as_edges_lengths()
print("")
graph.plot_A_star(start_node, end_node)
