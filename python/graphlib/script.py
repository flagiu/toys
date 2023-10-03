#!/bin/python3
from graphlib import Graph

graph = Graph()
try:
    graph = graph.load("graph.pickle")
except FileNotFoundError:
    N = 20
    #M = 30
    gamma = 1.5
    ok = False
    while not ok:
        graph.init_powerlaw(N,gamma)
        #graph.init_random(N,M)
        graph.dijkstra_all(0)
        ok = graph.is_connected
        if not ok:
            print("! graph is not connnected. Initializing again !")
    graph.relax()
    graph.save("graph.pickle")

start_node = 10
end_node = 18
graph.dijkstra_all(start_node)
print(f"graph.dijkstra_dist from node {start_node:d} =",graph.dijkstra_dist)
print(f"graph.dijkstra_prev from node {start_node:d} =",graph.dijkstra_prev)

graph.set_distance_as_edges_lengths()
graph.plot_dijkstra(start_node, end_node)
print("")
graph.plot_A_star(start_node, end_node)
