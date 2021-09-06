import time
import scipy
import pygad
import numpy as np
import networkx as nx
from numpy.random import default_rng
from simanneal import Annealer

graph16 = "0 4 0.19356591822011937 \
0 1 0.15308752612117948 \
1 5 0.9076949612383964 \
1 2 0.219827863804324 \
2 6 0.10808397800903757 \
2 3 0.09451217109601795 \
3 7 0.25085299097836455 \
4 8 0.2134874134102076 \
4 5 0.5227552788465125 \
5 9 0.3405283001156919 \
5 6 0.09270933806521578 \
6 10 0.1501483085272953 \
6 7 0.19526696582617503 \
7 11 0.3821347072953263 \
8 12 0.4458973081089166 \
8 9 0.1655806321013144 \
9 13 0.18763969728203858 \
9 10 0.4029882888425046 \
10 14 0.1510394942609051 \
10 11 0.12210502116514764 \
11 15 0.3645064594780176 \
12 13 0.34486162200143716 \
13 14 0.22646957055722094 \
14 15 0.8095775082535022"
graph25 = "0 5 0.19356591822011937 \
0 1 0.12210502116514764 \
1 6 0.9076949612383964 \
1 2 0.34486162200143716 \
2 7 0.10808397800903757 \
2 3 0.22646957055722094 \
3 8 0.25085299097836455 \
3 4 0.8095775082535022 \
4 9 0.2134874134102076 \
5 10 0.3405283001156919 \
5 6 0.22752208506704993 \
6 11 0.1501483085272953 \
6 7 0.09367835430750947 \
7 12 0.3821347072953263 \
7 8 0.09269572051388053 \
8 13 0.4458973081089166 \
8 9 0.11840905029654152 \
9 14 0.18763969728203858 \
10 15 0.1510394942609051 \
10 11 0.9666074742355028 \
11 16 0.3645064594780176 \
11 12 0.09613180025754518 \
12 17 0.15308752612117948 \
12 13 0.10301133353167141 \
13 18 0.219827863804324 \
13 14 0.11483240333668764 \
14 19 0.09451217109601795 \
15 20 0.5227552788465125 \
15 16 0.35858518328176875 \
16 21 0.09270933806521578 \
16 17 0.5012540554168993 \
17 22 0.19526696582617503 \
17 18 0.19435119476370893 \
18 23 0.1655806321013144 \
18 19 0.10146757967225838 \
19 24 0.4029882888425046 \
20 21 0.14747391328560275 \
21 22 0.119533977908775 \
22 23 0.30064237966083285 \
23 24 0.16035982688322076"
graph49 = "0 7 0.19356591822011937 \
0 1 0.1102384472645354 \
1 8 0.9076949612383964 \
1 2 0.3009255449829545 \
2 9 0.10808397800903757 \
2 3 0.10275486120756713 \
3 10 0.25085299097836455 \
3 4 0.3160751754093136 \
4 11 0.2134874134102076 \
4 5 0.11087713109253494 \
5 12 0.3405283001156919 \
5 6 0.1526518762702486 \
6 13 0.1501483085272953 \
7 14 0.3821347072953263 \
7 8 0.34986031494794356 \
8 15 0.4458973081089166 \
8 9 0.145220372411687 \
9 16 0.18763969728203858 \
9 10 0.16174964310633047 \
10 17 0.1510394942609051 \
10 11 0.0944588097486329 \
11 18 0.3645064594780176 \
11 12 0.7065179438743526 \
12 19 0.15308752612117948 \
12 13 0.37858957514542957 \
13 20 0.219827863804324 \
14 21 0.09451217109601795 \
14 15 0.09231115867854323 \
15 22 0.5227552788465125 \
15 16 0.10727255511942765 \
16 23 0.09270933806521578 \
16 17 0.39956452903350675 \
17 24 0.19526696582617503 \
17 18 0.3038470342324002 \
18 25 0.1655806321013144 \
18 19 0.1564606657415852 \
19 26 0.4029882888425046 \
19 20 0.38913408394253096 \
20 27 0.12210502116514764 \
21 28 0.34486162200143716 \
21 22 0.2359822275785209 \
22 29 0.22646957055722094 \
22 23 0.6697061367946509 \
23 30 0.8095775082535022 \
23 24 0.1232016723418686 \
24 31 0.22752208506704993 \
24 25 0.5600272271915072 \
25 32 0.09367835430750947 \
25 26 0.0913572187652071 \
26 33 0.09269572051388053 \
26 27 0.09798447517690209 \
27 34 0.11840905029654152 \
28 35 0.9666074742355028 \
28 29 0.11250759637110296 \
29 36 0.09613180025754518 \
29 30 0.11462554424803144 \
30 37 0.10301133353167141 \
30 31 0.21378836604797538 \
31 38 0.11483240333668764 \
31 32 0.12841814213639766 \
32 39 0.35858518328176875 \
32 33 0.11610912287479705 \
33 40 0.5012540554168993 \
33 34 0.17218334795861992 \
34 41 0.19435119476370893 \
35 42 0.10146757967225838 \
35 36 0.6190990307797452 \
36 43 0.14747391328560275 \
36 37 0.13772132788073566 \
37 44 0.119533977908775 \
37 38 0.23487364837795793 \
38 45 0.30064237966083285 \
38 39 0.14082740335857383 \
39 46 0.16035982688322076 \
39 40 0.19200179612781385 \
40 47 0.12355037763029052 \
40 41 0.09500909770171982 \
41 48 0.10812751882155199 \
42 43 0.331570905534615 \
43 44 0.11509382201909034 \
44 45 0.12522511475473558 \
45 46 0.15230830891484043 \
46 47 0.5554127606158855 \
47 48 0.3774272662281784"
g16 = graph16.split(" ")
adj = np.zeros((16, 16))
GA = False
for i in range(0, len(g16), 3):
    a = 1 / float(g16[i + 2])
    adj[int(g16[i])][int(g16[i + 1])] = a
G = nx.from_numpy_matrix(adj, create_using=nx.Graph())
ranges = dict(nx.all_pairs_dijkstra_path_length(G))

gene_space = range(len(adj))

max_load = 20 if len(adj) == 16 else 30

rng = default_rng()
numbers = rng.choice(len(adj), size=int(len(adj) / 4 + 1), replace=False)
depot = np.zeros(len(adj))
depot_idx = numbers[0]
depot[numbers[0]] = 1
demands = np.zeros(len(adj))
for i in range(int(len(adj) / 4)):
    demands[numbers[i + 1]] = np.random.randint(1, high=10) / max_load

if GA:
    def fitness(visits, solution_idx):
        dem = demands.copy()
        load = 1
        length = 0
        for index in range(1, len(visits)):
            if dem[int(visits[index])] < load:
                load -= dem[int(visits[index])]
                dem[int(visits[index])] = 0
                length += ranges[visits[index - 1]][visits[index]]
            else:
                length += ranges[visits[index - 1]][depot_idx]
                load = 1
                length += ranges[depot_idx][visits[index]]
                dem[int(visits[index])] = 0
        if dem.sum() > 0 or visits[-1] != depot_idx or visits[0] != depot_idx:
            length += dem.sum() * 5 * 1000 * len(adj)

        return -length


    start = time.time()
    solutions = []
    for h in range(2):
        numbers = rng.choice(len(adj), size=int(len(adj) / 4 + 1), replace=False)
        depot = np.zeros(len(adj))
        depot_idx = numbers[0]
        depot[numbers[0]] = 1
        demands = np.zeros(len(adj))
        for i in range(int(len(adj) / 4)):
            demands[numbers[i + 1]] = np.random.randint(1, high=10) / max_load
        # Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
        ga_instance = pygad.GA(num_generations=2000,
                               fitness_func=fitness,
                               num_parents_mating=10,
                               sol_per_pop=20,
                               gene_space=gene_space,
                               num_genes=len(adj) * 5)

        # Running the GA to optimize the parameters of the function.
        ga_instance.run()
        # ga_instance.plot_fitness()
        a = ga_instance.best_solution()
        solutions.append(a[1])
        end = time.time()
        print(h, "time elapsed:", end - start, "seconds")
    print(solutions)
    print(np.array(solutions).mean())
    print("done")
else:
    class VRP(Annealer):
        """Test annealer with a travelling salesman problem."""

        def __init__(self, state):
            #self.distance_matrix = distance_matrix
            super(VRP, self).__init__(state)  # important!
        def move(self):
            """Swaps two cities in the route."""
            self.state[np.random.randint(0, high=len(self.state))] = np.random.randint(0, high=len(adj))
            # a = random.randint(0, len(self.state) - 1)
            # b = random.randint(0, len(self.state) - 1)
            # self.state[a], self.state[b] = self.state[b], self.state[a]

        def energy(self):
            dem = demands.copy()
            load = 1
            length = 0
            for index in range(1, len(self.state)):
                if dem[int(self.state[index])] < load:
                    load -= dem[int(self.state[index])]
                    dem[int(self.state[index])] = 0
                    length += ranges[self.state[index - 1]][self.state[index]]
                else:
                    length += ranges[self.state[index - 1]][depot_idx]
                    load = 1
                    length += ranges[depot_idx][self.state[index]]
                    dem[int(self.state[index])] = 0
            if dem.sum() > 0 or self.state[-1] != depot_idx or self.state[0] != depot_idx:
                length += dem.sum() * 5 * 1000 * len(adj)

            return length

    start = time.time()
    solutions = []
    for h in range(16):
        numbers = rng.choice(len(adj), size=int(len(adj) / 4 + 1), replace=False)
        depot = np.zeros(len(adj))
        depot_idx = numbers[0]
        depot[numbers[0]] = 1
        demands = np.zeros(len(adj))
        for i in range(int(len(adj) / 4)):
            demands[numbers[i + 1]] = np.random.randint(1, high=10) / max_load

        initial_state = [depot_idx]
        for i in range(len(adj) * 5 - 2):
            initial_state.append(np.random.randint(0, high=len(adj)))
        initial_state.append(depot_idx)
        vrp = VRP(initial_state)
        itinerary, miles = vrp.anneal()
        print(miles)
        solutions.append(miles)
        end = time.time()
        print(h, "time elapsed:", end - start, "seconds")
    print(solutions)
    print(np.array(solutions).mean())
    print("done")
