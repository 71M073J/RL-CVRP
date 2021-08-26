"""Defines the main task for the VRP.

The VRP is defined by the following traits:
    1. Each city has a demand in [1, 9], which must be serviced by the vehicle
    2. Each vehicle has a capacity (depends on problem), the must visit all cities
    3. When the vehicle load is 0, it __must__ return to the depot to refill
"""
import random

import matplotlib
import networkx as nx
import nodevectors as nv
import numpy as np
import torch
import umap
import math
from node2vec import Node2Vec
from torch.utils.data import Dataset

matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

torch.set_default_tensor_type(torch.DoubleTensor)


class VehicleRoutingDataset(Dataset):
    def __init__(self, num_samples, graph_size, max_load=20, max_demand=9,
                 seed=None, embedding=None, neighbours=5, graf="manhattan", avg_power=5, num_demands=4, max_dist=10.,
                 different_num_of_demands=False, enc_feats=16):
        super(VehicleRoutingDataset, self).__init__()

        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        if seed is None:
            seed = np.random.randint(1234567890)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.isNewGraph = neighbours > 0
        self.num_samples = num_samples
        self.max_load = max_load
        self.max_demand = max_demand
        self.num_demands = num_demands

        torch.set_default_tensor_type(torch.DoubleTensor)
        # if graf == "manhattan":
        locations = []
        # Depot location will be the first node in each

        # TODO locations = torch.rand((num_samples, 2, graph_size + 1))

        # TODO self.locs = locations!!!!! to če že za različno generiranje grafa
        # start = time.time()
        # distances = torch.stack([
        #     torch.stack([
        #         torch.sqrt(torch.sum(torch.stack([torch.square(self.locs[:, 0, x] - self.locs[:, 0, y]),
        #                                           torch.square(self.locs[:, 1, x] - self.locs[:, 1, y])], dim=0), dim=0))
        #         for y in range(graph_size + 1)], dim=1) for x in range(graph_size + 1)], dim=1)

        # TODO edges = torch.randint(0, graph_size, (2, avg_power * graph_size))
        g = nx.Graph()
        g.add_nodes_from(range(graph_size))
        t = math.floor(math.sqrt(graph_size))
        g.add_weighted_edges_from(
            [(i * t + j, (i + 1) * t + j, random.random() * max_dist) for i in range(t - 1) for j in range(t)])
        g.add_weighted_edges_from(
            [(i * t + j, i * t + j + 1, random.random() * max_dist) for i in range(t) for j in range(t - 1)])
        adjacencies = torch.zeros((graph_size, graph_size,))  # adjacencies - the same for every sample
        for i, adj in g.adjacency():
            for node in adj:
                adjacencies[i, node] = g.edges[i, node]["weight"]
        self.adj = adjacencies.to(device)
        # to je static value enak za vse sample, ker treniramo na istem grafu
        # zato so to tudi node embeddingi, ker se parametri networka ne spreminjajo
        # self.road_lengths = torch.tensor((np.zeros([graph_size, graph_size]))) # costs of these roads, technically adjacencies not needed
        # features

        # TODO dynamic feature je tudi število visitov per-node

        if embedding == "node2vec":
            node2vec = Node2Vec(g, dimensions=enc_feats, walk_length=20, num_walks=200, workers=4)
            # Use temp_folder for big graphs

            # Embed nodes
            model = node2vec.fit()
            # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

            self.static = torch.tensor(model.wv.vectors).permute(1, 0).double()  # asf
        elif embedding == "GGVec":
            ggvec = nv.GGVec(n_components=enc_feats, order=2, max_epoch=1000, negative_ratio=0.5, learning_rate=0.05,
                             threads=4)
            embs = ggvec.fit_transform(g)
            self.static = torch.tensor(embs).permute(1, 0).double()
        elif embedding == "ProNE":
            prone = nv.ProNE(n_components=enc_feats)
            embs = prone.fit_transform(g)
            self.static = torch.tensor(embs).permute(1, 0).double()
        elif embedding == "GraRep":
            grarep = nv.GraRep(n_components=enc_feats)
            embs = grarep.fit_transform(g)
            self.static = torch.tensor(embs).permute(1, 0).double()
        elif embedding == "GLoVe":
            print("GLoVe se v smiselnem času ne odziva")
            exit(0)
            glove = nv.Glove(n_components=enc_feats, verbose=True, threads=4, max_epoch=1)
            embs = glove.fit_transform(g)
            self.static = torch.tensor(embs).permute(1, 0).double()
        elif embedding == "UMAP":
            embs = umap.UMAP(n_components=enc_feats, n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(
                adjacencies)
            self.static = torch.tensor(embs).permute(1, 0).double()
            # TODO DEJANSKE EMBEDDINGE HOLY FUCK SAMO TO MI JE BILO TREBA NAREST
        else:
            self.static = adjacencies

        # All states will broadcast the drivers current load
        # Note that we only use a load between [0, 1] to prevent large
        # numbers entering the neural network
        dynamic_shape = (num_samples, 1, graph_size)
        loads = torch.full(dynamic_shape, 1.)
        # All states will have their own intrinsic demand in [1, max_demand),
        # then scaled by the maximum load. E.g. if load=10 and max_demand=30,
        # demands will be scaled to the range (0, 3)
        # TODO
        # demands = torch.randint(1, max_demand + 1, dynamic_shape)
        # demands = demands / float(max_load)

        weights = torch.full((num_samples, graph_size), 1.)
        x = torch.multinomial(weights, 1 + num_demands, False)
        depot = torch.full((num_samples, graph_size), 0.)
        depot[range(num_samples), x[:, 0]] = 1
        self.depot = depot
        demand = torch.full(dynamic_shape, 0.)
        if different_num_of_demands:  # TODO not fully implemented
            for i in range(num_samples):
                x = torch.multinomial(torch.ones((graph_size,)),
                                      random.randint(1, graph_size - 2))  # TODO check if is depot, if so ignore
                for j in range(graph_size):
                    demand[range(num_samples), 0, x[:, j + 1]] = 1
        else:
            for i in range(0, num_demands):
                demand[range(num_samples), 0, x[:, i + 1]] = torch.randint(1, max_demand + 1, (num_samples,)) / float(
                    max_load)

        self.dynamic = torch.cat((loads, demand), dim=1)
        print("Data Generated")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, x,y coords)
        return (self.adj, self.static, self.dynamic[idx], self.depot[idx])

    def update_mask(self, depot, dynamic, chosen_idx=None):
        """Updates the mask used to hide non-valid states.
        """
        # TODO
        #with torch.no_grad():
        loads = dynamic[:, 0].detach()  # (batch_size, seq_len)
        demands = dynamic[:, 1].detach()  # (batch_size, seq_len)

        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        if demands.eq(0).all():  # if all demands are 0, kinda wack code
            return demands * 0.

        # We should avoid traveling to the depot back-to-back
        in_depot = torch.nonzero(depot)[:, 1].eq(chosen_idx)

        # ... unless we're waiting for all other samples in a minibatch to finish
        has_no_demand = demands[:, :].sum(dim=1).eq(0)

        new_mask = self.adj[chosen_idx].gt(0).double() * demands.le(loads)
        done = in_depot * has_no_demand
        new_mask[done, :] = depot[done, :]

        return new_mask.float()

    def update_dynamic(self, dynamic, chosen_idx, depot):
        """Updates the (load, demand) dataset values."""

        # Update the dynamic elements differently for if we visit depot vs. a city
        # Clone the dynamic variable so we don't mess up graph
        all_loads = dynamic[:, 0].clone()#.detach()
        all_demands = dynamic[:, 1].clone()#.detach()
        #with torch.no_grad():
        # loads = dynamic.data[:, 0]  # (batch_size, seq_len)
        # demands = dynamic.data[:, 1] #TODO

        # visit = torch.full((256,), 0)
        visit = torch.gather(all_demands.gt(0), 1, chosen_idx.unsqueeze(1))  # [range(chosen_idx.size(0)), chosen_idx]
        # demand_indices = torch.nonzero(all_demands)
        # for i in range(len(demand_indices)):
        #    if chosen_idx[demand_indices[i, 0]] == demand_indices[i, 1]:
        #        visit[demand_indices[i, 0]] = 1.

        depot_idx = torch.nonzero(depot)[:, 1]
        in_depot = depot_idx.eq(chosen_idx)

        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))

        # Across the minibatch - if we've chosen to visit a city, try to satisfy
        # as much demand as possible
        if visit.any():
            new_load = torch.clamp(load - demand, min=0)
            new_demand = torch.clamp(demand - load, min=0)

            # Broadcast the load to all nodes, but update demand seperately
            visit_idx = visit.squeeze().nonzero().squeeze()
            all_loads[visit_idx] = new_load[visit_idx]
            all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(
                -1)  # update visited nodes' demands
            # TODO A to updatanje loada dela prav???to dol
            all_demands[visit_idx, depot_idx[visit_idx]] = -1. + new_load[visit_idx].view(
                -1)  # set the demand of the depot to however we've spent, i.e. negative

        # Return to depot to fill vehicle load
        if in_depot.any():
            temp = in_depot.nonzero().squeeze()
            all_loads[temp] = 1.  # we refresh load and demand of the depot
            all_demands[temp, depot_idx[temp]] = 0.

        return torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1).clone().detach().requires_grad_(True)  # torch.tensor(tensor.data, device=dynamic.device)


def reward(dynamic, tour_indices, adj, depot, num_nodes):
    """
    Euclidean distance between all cities / nodes given by tour_indices
    """

    # Convert the indices back into a tour
    length = adj[0, tour_indices[:, 1:], tour_indices[:, :-1]].sum(1)

    #visit = torch.gather(all_demands.gt(0), 1, chosen_idx.unsqueeze(1))  # [range(chosen_idx.size(0)), chosen_idx]

    all_demands = dynamic[:, 1].clone()
    did_not_finish = tour_indices[:, -1].ne(torch.nonzero(depot)[:, 1]) * torch.sum(all_demands, dim=1).gt(0)
    length += did_not_finish.mul(all_demands.sum(dim=1).mul(num_nodes * 5 * 1000))

    # print(tour)
    # torch.set_printoptions(profile="default")

    # Ensure we're always returning to the depot - note the extra concat
    # won't add any extra loss, as the euclidean distance between consecutive
    # points is 0
    # start = static.data[:, :, 0].unsqueeze(1)
    # y = torch.cat((start, tour, start), dim=1)

    # Euclidean distance between each consecutive point
    # tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return length  # torch.tensor(length, device=device)


def render(static, tour_indices, save_path):
    """Plots the found solution."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        # Assign each subtour a different colour & label in order traveled
        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        where = np.where(idx == 0)[0]

        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)

        ax.legend(loc="upper right", fontsize=3, framealpha=0.5)
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)


'''
def render(static, tour_indices, save_path):
    """Plots the found solution."""

    path = 'C:/Users/Matt/Documents/ffmpeg-3.4.2-win64-static/bin/ffmpeg.exe'
    plt.rcParams['animation.ffmpeg_path'] = path

    plt.close('all')

    num_plots = min(int(np.sqrt(len(tour_indices))), 3)
    fig, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                             sharex='col', sharey='row')
    axes = [a for ax in axes for a in ax]

    all_lines = []
    all_tours = []
    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        cur_tour = np.vstack((x, y))

        all_tours.append(cur_tour)
        all_lines.append(ax.plot([], [])[0])

        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

    from matplotlib.animation import FuncAnimation

    tours = all_tours

    def update(idx):

        for i, line in enumerate(all_lines):

            if idx >= tours[i].shape[1]:
                continue

            data = tours[i][:, idx]

            xy_data = line.get_xydata()
            xy_data = np.vstack((xy_data, np.atleast_2d(data)))

            line.set_data(xy_data[:, 0], xy_data[:, 1])
            line.set_linewidth(0.75)

        return all_lines

    anim = FuncAnimation(fig, update, init_func=None,
                         frames=100, interval=200, blit=False,
                         repeat=False)

    anim.save('line.mp4', dpi=160)
    plt.show()

    import sys
    sys.exit(1)
'''
