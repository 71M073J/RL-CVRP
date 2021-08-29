"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""

import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ReinforceRL4CVRP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate(data_loader, actor, reward_fn, num_nodes, render_fn=None, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):
        adj, static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None
        adj = adj.to(device)

        with torch.no_grad():
            _, tour_indices, _ = actor.forward(adj, static, dynamic, x0)

        reward = reward_fn(static, tour_indices, adj, x0, num_nodes).mean().item()
        rewards.append(reward)

        # if render_fn is not None and batch_idx < num_plot:
        #    name = 'batch%d_%2.4f.png' % (batch_idx, reward)
        #    path = os.path.join(save_dir, name)
        #    render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards)


def train(actor, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, gamma, max_grad_norm,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    now = now + kwargs['emb']
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_params = None
    best_reward = np.inf

    for epoch in range(20):

        actor.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):

            adj, static, dynamic, x0 = batch

            adj = adj.to(device)
            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through the dataset
            dynamic, tour_indices, tour_logp = actor(adj, static, dynamic, x0)

            # Sum the log probabilities for each city in the tour
            reward = reward_fn(dynamic, tour_indices, adj, x0, num_nodes)

            actor_loss = torch.sum(reward * tour_logp.sum(dim=1))

            actor_optim.zero_grad()
            actor_loss.backward()
            # TODO? use or not
            # torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())
            if (batch_idx + 1) % 1 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                print('  Batch %d/%d, tour length: %2.3f, avg. loss: %2.4f, took: %2.4fs' %
                      (batch_idx + 1, len(train_loader), mean_reward, mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % epoch)

        mean_valid = validate(valid_loader, actor, reward_fn, num_nodes, render_fn,
                              valid_dir, num_plot=5)

        # Save best model parameters
        if mean_valid < best_reward:
            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs '
              '(%2.4fs / 100 batches)\n' %
              (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
               np.mean(times) * 100))


def train_vrp(args):
    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)
    import vrp
    from vrp import VehicleRoutingDataset

    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {16: 20, 49: 30, 1: 10, 25: 30, 36: 55, 64: 50, 81: 50, 5: 10, 10: 20, 20: 30, 50: 40, 100: 50}
    # TODO convert these values into self. attributes of VRPdataset
    MAX_DEMAND = 9
    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 2  # (load, demand)
    max_load = LOAD_DICT[args.num_nodes]
    enc_feats = 32
    STATIC_SIZE = enc_feats
    num_nodes = args.num_nodes
    embedding = "node2vec"
    train_data = VehicleRoutingDataset(args.train_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed,
                                       embedding,
                                       enc_feats=enc_feats)
    # print(train_data)
    # exit(0)
    # print("adf")
    valid_data = VehicleRoutingDataset(args.valid_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed + 1,
                                       embedding,
                                       enc_feats=enc_feats)

    actor = ReinforceRL4CVRP(STATIC_SIZE,
                             DYNAMIC_SIZE,
                             args.hidden_size,
                             train_data.update_dynamic,
                             train_data.update_mask,
                             args.num_layers,
                             args.dropout,
                             enc_feats,
                             num_nodes).to(device)

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = vrp.reward
    kwargs['render_fn'] = None  # TODO vrp.render
    kwargs['emb'] = embedding
    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, **kwargs)

    test_data = VehicleRoutingDataset(args.valid_size,
                                      args.num_nodes,
                                      max_load,
                                      MAX_DEMAND,
                                      args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, vrp.reward, num_nodes, vrp.render, test_dir, num_plot=5)

    print('Average tour length: ', out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_nodes', default=49, type=int)
    parser.add_argument('--actor_lr', default=5e-3, type=float)
    parser.add_argument('--gamma', default=0.995, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size', default=65536, type=int)  # 65536
    parser.add_argument('--valid-size', default=256, type=int)

    args = parser.parse_args()

    # print('NOTE: SETTTING CHECKPOINT: ')
    # args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
    # print(args.checkpoint)

    if args.task == 'vrp':
        train_vrp(args)
    else:
        raise ValueError('Task <%s> not understood' % args.task)
