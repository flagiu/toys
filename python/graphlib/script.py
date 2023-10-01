from graphlib import Graph

graph = Graph()
N = 40
M = 30
gamma = 2 #1.5
ok = False
while not ok:
    graph.init_powerlaw(N,gamma)
    #graph.init_random(N,M)
    graph.dijkstra_all(0)
    ok = graph.is_connected
    if not ok:
        print("! graph is not connnected. Initializing again !")
print("graph.dijkstra_dist =",graph.dijkstra_dist)
print("graph.dijkstra_prev =",graph.dijkstra_prev)
graph.relax()
print("graph.dijkstra(0,1) =",*graph.dijkstra(0,1))
graph.save("graph.pickle")
graph.plot()