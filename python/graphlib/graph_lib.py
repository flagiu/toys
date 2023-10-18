import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from functools import partial
import pickle
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10

class Node():
    def __init__(
        self, mass = 1.0, neigh = [], label = None
    ):
        """
        :param list[int] neigh
        """
        self.mass = mass
        self.neigh = neigh
        self.label = label
        return
    
    def degree(self):
        return len(self.neigh)
    
    def print(self):
        print(f"( {self.label} ) --> (",end='')
        for n in self.neigh:
            print(f" {n}",end='')
        print(" )")
        return

#----------------------------------------------------------------------------------#

class Edge():
    def __init__(
        self, start: int, end: int, directed = False, weight = 1.0,
        length = 1.0, length_label_frac=0.3, stiffness = 1, x0 = 1., damping = 1.
    ):
        self.start = start
        self.end = end
        self.directed = directed
        self.weight = weight
        self.length = length
        self.length_label_frac = length_label_frac
        self.stiffness = stiffness
        self.x0 = x0
        self.damping = damping
        return
    
    def same(self, other):
        sameParallel = (self.start == other.start and self.end == other.end)
        sameOpposite = (self.start == other.end and self.end == other.start)
        if self.directed or other.directed:
            return sameParallel
        else:
            return sameParallel or sameOpposite
    
    def print(self):
        if self.directed:
            tag = '-->'
        else:
            tag = '---'
        print(self.start,tag,self.end)
        return

#----------------------------------------------------------------------------------#

class Graph():
    def __init__(
        self, nodes: tuple[Node] = None, edges: tuple[Edge] = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self.A = None #adjacency matrix
        self.R = 2.0
        self.x = None
        if nodes is not None and self.edges is not None:
            self.get_adjacency_matrix()
            self.init_neigh()
            self.init_positons()
        return
    
    def __len__(self):
        return len(self.nodes)
    
    def connectivity(self):
        return len(self.edges)
    
    def get_adjacency_matrix(self):
        A = np.zeros( (len(self.nodes),len(self.nodes)), dtype=int )
        for edge in self.edges:
            A[ edge.start, edge.end ] = 1
            if not edge.directed:
                A[ edge.end, edge.start ] = 1
        self.A = A
        return A.copy()

    def init_neigh(self):
        for node in self.nodes:
            node.neigh = []
        for edge in self.edges:
            s,e = (edge.start, edge.end)
            self.nodes[s].neigh += [e]
            if not edge.directed:
                self.nodes[e].neigh += [s]
        return
    
    def get_edge(self, start, end):
        myEdge = Edge(start,end)
        for edge in self.edges:
            if edge.same(myEdge):
                return edge
    
    def init_positions(self, layout="circle"):
        if layout=="circle":
            self.init_positions_circle( len(self)/(2*3.141592) )
            print("\nGraph.init_positions(): positions inizialized on a circle.")
        else:
            print(f"[ Error: layout {layout} not valid. ]")
        return
    
    def init_positions_circle(self, R=2.0):
        N = len(self)
        self.x = np.empty((N,2), dtype=np.float32)
        self.R = R
        angles = np.linspace(0, 2*np.pi*(1-1/N), N)
        self.x[:,0] = R*np.cos(angles)
        self.x[:,1] = R*np.sin(angles)
        return
    
    def set_gaussian_edges_x0(self, mean=1.0, sigma=0.2):
        randoms = np.random.randn(len(self.edges))
        randoms = (randoms+mean)*sigma
        for i,edge in enumerate(self.edges):
            edge.x0 = max(randoms[i], 0.01)
        return
    
    def set_constant_edges_lengths(self, value=1.0):
        for edge in self.edges:
            edge.length = value
        return
    
    def set_distance_as_edges_lengths(self):
        def distance(x1,x2):
            return np.sum((x1-x2)**2)**0.5
        for edge in self.edges:
            edge.length = distance(self.x[edge.start], self.x[edge.end])
        return

    def init_random(self, N, M, maxcount=100000):
        print(f"\rGraph.init_random(): I will initialize the graph uniformly at random.")
        if M>N*(N-1)/2:
            print("[ERROR: too many connections]")
            return
        self.nodes = [Node(label=str(i)) for i in range(N)]
        self.edges = []
        for edge_count in range(M):
            exists = True
            count = 0
            while exists and count<maxcount:
                count += 1
                k = np.random.randint(N*N)
                i = int(k%N)
                j = int(k/N)
                if i==j:
                    continue
                if i<j:
                    (i,j) = (j,i)
                #i = np.random.randint(N)
                #j = np.random.randint(i+1,i+1+(N-1))%N #different from i
                newEdge = Edge(i,j)
                exists = False
                for oldEdge in self.edges:
                    if oldEdge.same(newEdge):
                        exists = True
                        #print('exists!')
                        break
            if count>=maxcount:
                print(f"[ ERROR: too many iterations for N={N}, M={M} ]")
                return
            self.edges += [ newEdge ]
            print(f"\rGraph.init_random(): Connected {1+edge_count}/{M} edges", end='')
        print("")
        self.init_neigh()
        self.get_adjacency_matrix()
        return
    
    def init_powerlaw(self, N, gamma, maxcount=10000):
        print(f"\rGraph.init_powerlaw(): I will initialize the graph at random using a power law ~1/k^gamma (k = node's degree; gamma = {gamma:.2f}).")
        if gamma<=1:
            print("[ERROR: gamma must be > 1]")
            return
        self.nodes = [Node(label=str(i)) for i in range(N)]
        possible = False
        count = 0
        while (not possible) and (count<maxcount):
            # Extract all nodes' degree in one shot
            r = np.random.rand(N)
            k = (1-(1-1/N**(gamma-1))*r)**(-1/(gamma-1)) # distributed as 1/k**gamma, 1<=k<= N
            k = np.floor(k+0.5) #round to integer
            #print("extracted k = ",k)
            # Try to connect nodes
            possible = self.connect_nodes(k)
            count += 1
            print(f"\rGraph.init_powerlaw(): Attempt {count}/{maxcount}", end='')
        print("")
        if count>=maxcount:
            print(f"[ ERROR: could not initialize scale-fre graph with gamma={gamma} ]")
            print(f"[ Last sample of degrees: {k} ]")
        self.init_neigh()
        self.get_adjacency_matrix()
        return
    
    def init_cayley_tree(self, degree: int, levels: int, maxcount=10000):
        print(f"\rGraph.init_cayley_tree(): I will initialize the graph as a Cayley tree with degree {degree} and {levels} levels.")
        if degree<2:
            print("[ERROR: degree must be >= 3]")
            return
        elif degree>2:
            N = int(1+degree/(degree-2)*( (degree-1)**levels -1) )
        else:
            N = 1+degree*levels
        self.nodes = [Node(label=str(i)) for i in range(N)]
        self.edges = []
        old_frontier=[0]
        x0=1.0
        for level in range(1,levels+1): # level = 2,..,levels
            num_new_connections = degree if level==1 else degree-1
            len_new_frontier = num_new_connections*len(old_frontier)
            new_frontier=[ i for i in range(old_frontier[-1]+1, old_frontier[-1]+1 + len_new_frontier ) ]
            for i,node_idx in enumerate(old_frontier):
                for di in range(num_new_connections):
                    self.edges.append( Edge(node_idx, new_frontier[num_new_connections*i+di], x0=x0) )
            old_frontier = new_frontier
            x0/=1.5
        self.init_neigh()
        self.get_adjacency_matrix()
        return
    
    def init_regular(self, N: int, degree: int, maxcount=10000):
        print(f"\rGraph.init_cayley_tree(): I will initialize the graph at random using the same degree k={degree} for each node.")
        if degree<=2:
            print("[ERROR: degree must be >= 2]")
            return
        self.nodes = [Node(label=str(i)) for i in range(N)]
        possible = False
        count = 0
        while (not possible) and (count<maxcount):
            k = np.ones(N)*degree
            # Try to connect nodes
            possible = self.connect_nodes(k)
            count += 1
            print(f"\rGraph.init_regular(): Attempt {count}/{maxcount}", end='')
        print("")
        if count>=maxcount:
            print(f"[ ERROR: could not initialize regular graph with degree k={degree} ]")
        self.get_adjacency_matrix()
        return
    
    def connect_nodes(self, degrees, maxcount=10000):
        # Starting from the degree of each node, try to connect between them if possible
        if (int(np.sum(degrees))%2 == 0): # must be even
            return False
        N = len(self.nodes)
        self.edges = []
        for node in self.nodes:
            node.neigh = []
        for i in range(N-1):
            n = self.nodes[i]
            # For as many neighbours as you need:
            for j in range(int(degrees[i]) - len(n.neigh)):
                match=False
                count = 0
                # Try to find a neighbour:
                while (not match) and (count < maxcount):
                    k = ( i+1 + np.random.randint(N-1) )%N
                    # Not possible if k has already a max num of neighbours
                    if len(self.nodes[k].neigh) >= int(degrees[k]):
                        match=False
                    # Else, check if you already connected to it
                    elif sum([ nn==k for nn in n.neigh ]):
                        match=False
                    # Else OK
                    else:
                        n.neigh.append( k )
                        self.nodes[k].neigh.append( i )
                        self.edges.append( Edge(i,k) )
                        match=True
                    count += 1
                #print(f"node {i} neigh n.{j+1}: {count} attempts")
                # Could not find it in a reasonable time
                if count>=maxcount:
                    return False
        # Manually add an extra node to match the degree of the last node
        self.nodes[-1].neigh.append( N )
        self.nodes.append( Node(label=str(N)) )
        self.nodes[-1].neigh = [N-1]
        self.edges.append( Edge(N-1,N) )
        return True
    
    #--------------------------------------------------------------------------------#

    def relax(self, dt=0.01, tolerance=0.0001, maxcount=80000, output_every=200,
              wall_damping=.99, repulsion_stiffness=50, crunchMode = None
        ):
        print(f"\rGraph.relax(): I will relax nodes' position using a force-based method.")
        self.init_positions()
        N = len(self)
        invN = 1/N
        m = np.array([n.mass for n in self.nodes], dtype=np.float32)
        m = m[:,None]
        x = self.x.copy()
        L_fin = 4*2*np.array([[self.R,self.R]], dtype=np.float32)
        L_in = 2*L_fin
        L_tau = 0.1
        if crunchMode==None:
            L_schedule = [L_fin]*maxcount
        elif crunchMode=="lin":
            L_schedule = [ L_fin*c/float(maxcount) + (1-c/float(maxcount))*L_in for c in range(maxcount) ]
        elif crunchMode=="exp":
            L_schedule = [ L_fin + (L_in-L_fin)*np.exp(-c/float(maxcount) /L_tau) for c in range(maxcount) ]
        else:
            print(f"Graph.relax(): ERROR: crunch mode '{crunchMode}' not recognized.\n")
            sys.exit(1)
        np.savetxt("L_schedule.dat", np.array(L_schedule)[:,:,0] )
        print(f"\rGraph.relax():  L schedule saved into L_schedule.dat ")
        dx = np.empty((N,2), dtype=np.float32)
        v = np.zeros((N,2), dtype=np.float32)
        maxdx = 10
        count = 0
        X = [x]
        maxdx_history=[]
        while maxdx > tolerance and count<maxcount:
            L = L_schedule[count]
            f = np.zeros((N,2), dtype=np.float32) #forces
            # Bonded forces
            for edge in self.edges:
                i,j = (edge.start, edge.end)
                rij = x[i]-x[j]
                f[i] += ( -edge.stiffness*( rij-edge.x0) - edge.damping*v[i] )
                f[j] += ( -edge.stiffness*(-rij-edge.x0) - edge.damping*v[j] )
            # 1/r^6 repulsion between ALL nodes
            all_nodes_list = [i for i in range(N)]
            for i in range(N):
                nonneigh = list(set(all_nodes_list) - set([i]))#+self.nodes[i].neigh))
                for j in nonneigh:
                    xij = (x[i]-x[j])
                    rij = np.sum(xij*xij)**(6/2)
                    if rij>tolerance*10:
                        f[i] += repulsion_stiffness *xij/rij
                        f[j] -= repulsion_stiffness *xij/rij
            # simple Euler integration
            v += f/m*dt 
            # remove c.o.m. velocity
            vCM = np.sum(v*m, axis=0)/np.sum(m)
            v -= vCM
            dx = v*dt
            maxdx = np.max( np.abs(dx) )
            x += dx
            # Cubic boundary condition
            """
            for dim in range(2):
                mask = x[:,dim]>0.5*L[0,dim]
                x[mask, dim] = 0.5*L[0,dim]
                v[mask, dim] *= -wall_damping
                mask = x[:,dim]<-0.5*L[0,dim]
                x[mask, dim] = -0.5*L[0,dim]
                v[mask, dim] *= -wall_damping
            """
            # Spherical boundary condition: it conserves total angular momentum (?)
            r_sbc = float(0.5*np.min(L))
            r_all = np.sqrt(x[:,0]**2+x[:,1]**2)
            angle_all = np.arctan2(x[:,1],x[:,0])
            mask = r_all > r_sbc
            num_sbc = np.sum(mask)
            if(num_sbc>0):
                x[mask,0] = (r_sbc-tolerance) * np.cos(angle_all[mask])
                x[mask,1] = (r_sbc-tolerance) * np.sin(angle_all[mask])
                v[mask] *= -wall_damping
                #print(f"\rGraph.relax(): applied spherical BC to {num_sbc} atoms at time {count}.",end="")
            
            count += 1
            maxdx_history.append(maxdx)
            if count%output_every ==0:
                print(f"\rGraph.relax():  [Iteration = {count}/{maxcount}], [Position_update = {maxdx:.5f}/{tolerance:.5f}]", end='')
                X.append(x.copy())
        
        self.x = x.copy()
        self.set_distance_as_edges_lengths()
        print(f"\rGraph.relax():  [Iteration = {count}/{maxcount}], [Position_update = {maxdx:.5f}/{tolerance:.5f}]")
        np.savetxt("maxdx.dat", np.array(maxdx_history).T )
        print(f"\rGraph.relax():  maximum displacement among nodes saved into maxdx.dat ")
        return X, L
    
    def get_edge_lines(self, x):
        lx = [] #edge of segment
        ly = []
        cx = [] #center of segment
        cy = []
        lengths = []
        for edge in self.edges:
            s,e = (edge.start, edge.end)
            lx.append( x[s,0] )
            lx.append( x[e,0] )
            lx.append( None )
            ly.append( x[s,1] )
            ly.append( x[e,1] )
            ly.append( None )
            cx.append( (edge.length_label_frac*x[s,0]+(1-edge.length_label_frac)*x[e,0]) )
            cy.append( (edge.length_label_frac*x[s,1]+(1-edge.length_label_frac)*x[e,1]) )
            lengths.append(f"{edge.length:.2f}")
        return (lx,ly),(cx,cy), lengths
    
    def plot(self, fig=None, ax=None, outpng="graphlib_plot.png"):
        x = self.x
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        ax.axis('equal')
        ax.plot(x[:,0],x[:,1],'ro')
        for i in range(len(self)):
            ax.annotate(self.nodes[i].label, (x[i,0],x[i,1]), fontsize=12)
        
        (lx,ly),(cx,cy),dists = self.get_edge_lines(x)
        ax.plot(lx,ly, 'k--', alpha=0.3)
        for Cx,Cy,dist in zip(cx,cy,dists):
            ax.annotate(dist, (Cx,Cy), ha='center', va='center', color='blue', alpha=0.5, fontsize=8)
        plt.axis('off')
        plt.savefig(outpng, bbox_inches='tight')
        print(f"Graph.plot(): Figure saved into {outpng} ")
        #plt.show()
        return fig, ax
    
    def save_relax_gif(self, X,L):
        L = L.T
        nevery = int(1)
        for frame,x in enumerate(X):
            fig, ax = plt.subplots(figsize=(4,4))
            ax.plot(x[:,0], x[:,1], 'ro')
            for i in range(len(self)):
                ax.annotate(self.nodes[i].label, (x[i,0],x[i,1]))
            (lx,ly),(cx,cy),dists = self.get_edge_lines(x)
            for Cx,Cy,dist in zip(cx,cy,dists):
                ax.annotate(dist, (Cx,Cy), ha='center', va='center', color='blue', alpha=0.5)
            ax.plot(lx,ly, [], 'k', alpha=0.7)
            ax.set_title(f"frame {frame}")
            ax.set_xlim( (-0.5*L[0], 0.5*L[0]))
            ax.set_ylim( (-0.5*L[1], 0.5*L[1]))
            ax.axis('equal')
            #fig.savefig(f"graphlib_gif_{frame:04d}.png", bbox_inches='tight')
        print(f"Graph.plot(): Figures saved into graphlib_gif_XXXX.png ")
        return

    def animate(self):
        X, L = self.relax()
        L = L.T
        nevery = int(1)
        fig, ax = plt.subplots(figsize=(4,4))
        plt.rcParams["animation.html"] = "jshtml"
        plt.rcParams['figure.dpi'] = 150  
        plt.ioff()
        line1, = ax.plot([], [], 'ro')
        line2, = ax.plot([], [], 'k', alpha=0.7)
        texts = [matplotlib.text.Text() for n in range(len(self))]
        def init():
            ax.set_xlim( (-0.5*L[0], 0.5*L[0]))
            ax.set_ylim( (-0.5*L[1], 0.5*L[1]))
            ax.axis('equal')
            #ax.axis('off')
            return line1,line2
        def update(frame, ln1, ln2, txts, X):
            x = X[frame]
            ln1.set_data(x[:,0], x[:,1])
            lx,ly = self.get_edge_lines(x)
            ln2.set_data(lx, ly)
            for i in range(len(self)):
                #ax.text(x[i,0]+0.01,x[i,1]+0.01, self.nodes[i].label)
                dic = {'x': x[i,0]+0.01, 'y':x[i,1]+0.01, 'text': self.nodes[i].label }
                txts[i].update(dic)
            plt.title(f"frame {frame}")
            return ln1,ln2
        
        return matplotlib.animation.FuncAnimation(fig, partial(update, ln1=line1, ln2=line2, txts=texts, X=X),
                                                  frames=range(len(X)), init_func=init, blit=True)
    
    def print(self):
        print("\nNodes:")
        if self.nodes is not None:
            [ n.print() for n in self.nodes ]
        print("Edges:")
        if self.edges is not None:
            [ e.print() for e in self.edges ]
        if self.A is not None:
            print("Adjacency matrix:")
            print(self.A)
        return
    
    def save(self, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        print(f"\nGraph.save(): Graph saved to {filename} .")
    
    def load(self, filename):
        with open(filename, 'rb') as inp:
            obj = pickle.load(inp)
        return obj
            
    #----------------------------------------------------------------------------#
    #---------- PATH FINDING algorithms -----------------------------------------#
    #----------------------------------------------------------------------------#
    def dijkstra(self, idx_of_source_node: int, idx_of_target_node: int):
        """
        # This algorithm finds the paths of minimum GRAPH distance from the source node to the target node.
        """
        infinity = 9999999999
        dist = [infinity] * len(self)       # list of minimum distance from source to each node
        prev = [None] * len(self)           # list of previous step in the path from source to each node
        Q = [ i for i in range(len(self))]  # list of all node indexes
        Q_init = np.array(Q, dtype=int)
        dist[idx_of_source_node] = 0
        while len(Q)>0:
            idx_u = 0
            u = Q[idx_u]
            for idx_u_, u_ in enumerate(Q):
                if dist[u_]<dist[u]:
                    u = u_
                    idx_u = idx_u_
            # Now u is the index of closest node in Q to source.
            # Remove it from Q
            Q.pop(idx_u)
            if u == idx_of_target_node:
                break
            # Scan all neighbours of u which are still in Q
            u_neigh_in_Q = list(set( self.nodes[u].neigh ) & set(Q))
            for v in u_neigh_in_Q:
                alt = dist[u] + self.get_edge(u,v).length
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
        shortest_path = []
        shortest_path_length = 0.0
        u = idx_of_target_node
        if (prev[u] is not None) or (u==idx_of_source_node):
            while u is not None:
                shortest_path.insert(0, u)
                if prev[u] is not None:
                    shortest_path_length += self.get_edge(prev[u],u).length
                u = prev[u]
        return shortest_path, shortest_path_length

    def plot_dijkstra(self, idx_of_source_node: int, idx_of_target_node: int, outpng="dijkstra.png"):
        nodes_path, min_dist = self.dijkstra(idx_of_source_node, idx_of_target_node)
        labels_path = [ self.nodes[n].label for n in nodes_path ]
        print(f"Graph.plot_dijkstra(): Path from node {idx_of_source_node} to {idx_of_target_node}:")
        print(f"Graph.plot_dijkstra():  ",labels_path)
        print(f"Graph.plot_dijkstra():   distance = {min_dist:.2f}")
        fig, ax = self.plot(outpng=outpng)
        x_path = [ self.x[n,0] for n in nodes_path ]
        y_path = [ self.x[n,1] for n in nodes_path ]
        ax.plot(x_path, y_path, 'green')
        fig.savefig(outpng, bbox_inches='tight')
        print(f"Graph.plot_dijkstra(): Figure saved into {outpng}")
        return fig, ax
    
    def dijkstra_all(self, idx_of_source_node: int):
        """
        # This algorithm finds the paths of minimum GRAPH distance from the source node to each node in the graph.
        """
        infinity = 9999999999
        dist = [infinity] * len(self)       # list of minimum distance from source to each node
        prev = [None] * len(self)           # list of previous step in the path from source to each node
        Q = [ i for i in range(len(self))]  # list of all node indexes
        dist[idx_of_source_node] = 0
        while len(Q)>0:
            idx_u = 0
            u = Q[idx_u]
            for idx_u_, u_ in enumerate(Q):
                if dist[u_]<dist[u]:
                    u = u_
                    idx_u = idx_u_
            # Now u is the index of closest node in Q to source.
            # Remove it from Q
            Q.pop(idx_u)
            # Scan all neighbours of u which are still in Q
            u_neigh_in_Q = list(set(self.nodes[u].neigh) & set(Q))
            for v in u_neigh_in_Q:
                alt = dist[u] + self.get_edge(u,v).length
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
        self.dijkstra_dist = dist
        self.dijkstra_prev = prev
        self.is_connected = True
        for idx,p in enumerate(prev):
            if idx!=idx_of_source_node and (p is None):
                self.is_connected = False
                break
        return

    def A_star(self, idx_of_source_node: int, idx_of_target_node: int):
        """
        # This algorithm finds the paths of minimum REAL distance from the source node to the target node.
        """
        # PART 1: Define a potential: Euclidean distance from target node
        def potential(node_idx):
            return sum( ( self.x[node_idx] - self.x[idx_of_target_node])**2 )**0.5
        # PART 2: Potential reweighting of the edges' length
        for edge in self.edges:
            u = edge.start
            v = edge.end
            edge.length += potential(v) - potential(u)
        # PART 3: Run Dijkstra on the reweighted graph
        min_path, min_dist = self.dijkstra(idx_of_source_node, idx_of_target_node)
        # PART 4: Put back original lengths
        for edge in self.edges:
            u = edge.start
            v = edge.end
            edge.length += potential(u) - potential(v)
        real_min_dist = 0.0
        for i in range(len(min_path)-1):
            real_min_dist += self.get_edge(min_path[i],min_path[i+1]).length
        return min_path, real_min_dist
    
    def plot_A_star(self, idx_of_source_node: int, idx_of_target_node: int, outpng="A_star.png"):
        nodes_path, min_dist = self.A_star(idx_of_source_node, idx_of_target_node)
        labels_path = [ self.nodes[n].label for n in nodes_path ]
        print(f"Graph.plot_A_star(): Path from node {idx_of_source_node} to {idx_of_target_node}:")
        print(f"Graph.plot_A_star():  ",labels_path)
        print(f"Graph.plot_A_star():   distance = {min_dist:.2f}")
        fig, ax = self.plot(outpng=outpng)
        x_path = [ self.x[n,0] for n in nodes_path ]
        y_path = [ self.x[n,1] for n in nodes_path ]
        ax.plot(x_path, y_path, 'green')
        fig.savefig(outpng, bbox_inches='tight')
        print(f"Graph.plot_A_star(): Figure saved into {outpng}")
        return fig, ax
    
    #----------------------------------------------------------------------------#
    #---------- FLOOD FILL algorithms -------------------------------------------#
    #----------------------------------------------------------------------------#
    def flood_fill(self, idx_of_source_node: int):
        list_of_filled_node_idxs = [idx_of_source_node]
        def recursive_fill(lofni, node_idx):
            if not node_idx in lofni:
                return lofni
        ...
        return list_of_filled_node_idxs
    
    #----------------------------------------------------------------------------#
    #---------- MESSAGE PASSING algorithms --------------------------------------#
    #----------------------------------------------------------------------------#
    def as_factor_graph(self):
        factor_graph = Graph()
        for edge in self.edges:
            s = edge.start