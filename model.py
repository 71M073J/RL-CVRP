import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


class PolicyNetwork(nn.Module):
    def __init__(self, hidden_size, num_layers=1, dropout=0.2, enc_feats=-1, num_nodes=-1):
        super(PolicyNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * enc_feats),
                                          device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(2 + 2 * enc_feats + num_nodes, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = Attention(hidden_size, enc_feats, num_nodes)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, last_action, adj, static, dynamic, last_hh):
        last_action = last_action.expand(-1, -1, adj.size(1))
        rnn_in = torch.cat((last_action, adj, static, dynamic), dim=1).transpose(2, 1)
        rnn_out, last_hh = self.gru(rnn_in, last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh)

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(last_action, adj, static, dynamic, rnn_out)
        context = enc_attn.bmm(static.permute(0, 2, 1))  # (B, 1, num_feats) feature quality of encoded stuff?

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static)
        energy = torch.cat((static, context),
                           dim=1)  # (B, num_feats, seq_len) join static hidden state with current feature context

        v = self.v.expand(static.size(0), -1, -1)
        W = self.W.expand(static.size(0), -1, -1)
        # test = torch.bmm(W, energy)
        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)  # probabilities of node transition

        return probs, last_hh  # last hh is the last hidden state of the RNN


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size, enc_feats, num_nodes):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, hidden_size + 2 + 2 * enc_feats + num_nodes),
                                          device=device, requires_grad=True))

    def forward(self, last_action, adj, static, dynamic, rnn_out):
        rnn_out = rnn_out.transpose(2, 1)
        hidden = torch.cat((last_action, adj, static, dynamic, rnn_out), 1)

        batch_size, num_feats, num_nodes = hidden.size()
        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, num_nodes, -1)
        W = self.W.expand(batch_size, -1, -1)
        # test = torch.bmm(W, hidden)
        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)  # (batch, seq_len) - here we get per-node attentions for the entire batch
        return attns


class ReinforceRL4CVRP(nn.Module):
    """
    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
        (e.g. 2 for the VRP which has (load, demand) attributes. The TSP doesn't
        have dynamic elements, but to ensure compatility with other optimization
        problems, assume we just pass in a vector of zeros.
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """

    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.1, enc_feats=16, num_nodes=16):
        super(ReinforceRL4CVRP, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')
        self.agent = PolicyNetwork(hidden_size, num_layers, dropout, enc_feats, num_nodes)
        self.update_fn = update_fn
        self.mask_fn = mask_fn

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    def forward(self, adj, static, dynamic, depot, last_action=None, last_hh=None):
        """
        Parameters
        ----------
        adj: Array of size (num_nodes, num_nodes), representing the adjacency matrix
        static: Array of size (batch_size, feats, num_nodes)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, dynamic_feats, num_nodes)
            Defines the elements to consider as non-static. For the VRP, these are
            (load, demand) of each city.
        last_action: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        depot: depot
        """

        batch_size, input_size, num_nodes = static.size()

        if last_action is None:
            last_action = self.x0.expand(batch_size, -1, -1)

        # Always use a mask - if no function is provided, we don't update it
        mask = depot.clone().detach().requires_grad_(True) # torch.tensor(depot, device=device)  # torch.ones(batch_size, num_nodes, device=device)

        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        max_steps = num_nodes * 5 #TODO maybe uncomment? for big networks if self.mask_fn is None else 1000

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.

        for _ in range(max_steps):

            if not mask.byte().any():
                break

            # ... but compute a hidden rep for each element added to sequence
            #TODO še last action je zdej kot input
            probs, last_hh = self.agent(last_action, adj, static, dynamic, last_hh)
            # torch.set_printoptions(profile="full")
            #
            # print(probs + mask.log())
            # print("----------------------------------------------")
            # print(F.softmax(probs + mask.log()))
            # print("----------------------------------------------")
            # torch.set_printoptions(profile="default")
            probs = F.softmax(probs + mask.log(), dim=1)

            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            if self.training:

                m = torch.distributions.Categorical(probs)

                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                ptr = m.sample()  # choose from probs where to go next
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, ptr.data, depot)  # update loads/demands for visited node

                # Since we compute the VRP in minibatches, some tours may have a different
                # number of stops. We force the vehicles to remain at the depot 
                # in these cases, and logp := 0
                is_done = dynamic[:, 1].sum(1).eq(0).float()  # if sum of demands is equal to 0
                logp = logp * (1. - is_done)  # then logp is also 0, and we don't extend the tour

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                mask = self.mask_fn(depot, dynamic, ptr.data).detach()

            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))

            last_action = torch.gather(static, 2,
                                       ptr.view(-1, 1, 1)
                                       .expand(-1, input_size, 1)).detach()

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return dynamic, tour_idx, tour_logp


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
