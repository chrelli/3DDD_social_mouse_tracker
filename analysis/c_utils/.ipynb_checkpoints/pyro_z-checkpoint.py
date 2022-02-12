import os
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.infer.autoguide import AutoDiagonalNormal,init_to_sample
from pyro.ops.indexing import Vindex

pyro.enable_validation()
pyro.set_rng_seed(1987)


# %matplotlib inline
# for math and plotting
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#from tqdm.notebook import tqdm
from tqdm import tqdm

import torch

import os
from collections import defaultdict
import scipy.stats

import time as tm

def fit_pyro_z(data,data_dim = 1,hidden_dim = 2,show_plots = False, verbose = False):
    # for initializing, we use k-means, should we use the most likely state as first?
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=hidden_dim, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(data.unsqueeze(1).numpy())
    mu_prior = torch.tensor(kmeans.cluster_centers_,dtype=torch.float).squeeze()
    std_prior = torch.zeros(hidden_dim)
    for i in range(hidden_dim):
        std_prior[i] = torch.std(data[pred_y == i])
    print("prior mu: {} and std: {}".format(mu_prior,std_prior))

    if show_plots:
        plt.hist(data.numpy(),100)
        for i in range(hidden_dim):
            plt.axvline(mu_prior[i], c='r')
        plt.show()
        print("K-means prior is {}, shape is KxD: {}".format(mu_prior,mu_prior.shape))

    def my_init_fn(site):
        if site["name"] == "locs":
            return mu_prior
        if site["name"] == "scales":
            return std_prior
        # IF the site is not caugth above, we just init to a random sample from the prior
        return init_to_sample(site)



    @config_enumerate
    def hmm_model(data, data_dim=data_dim, hidden_dim=hidden_dim):
        if verbose:
            print('Running for {} time steps'.format(len(data)))

        # Sample global matrices wrt a Jeffreys prior.
        with pyro.plate("hidden_state", hidden_dim):
            transition = pyro.sample("transition", dist.Dirichlet(0.5 * torch.ones(hidden_dim)))

        locs = pyro.sample('locs', dist.Normal(torch.zeros(hidden_dim), 1.* torch.ones(hidden_dim) ) .independent() )
        scales = pyro.sample('scales', dist.LogNormal(torch.ones(hidden_dim)*-2,torch.ones(hidden_dim)*1 ) .independent() )
        if verbose:
            print(locs)
            print(scales)

        x = 0  # initial state
        for t, y in pyro.markov(enumerate(data)):
            x = pyro.sample("x_{}".format(t), dist.Categorical(transition[x]),
                            infer={"enumerate": "parallel"})
            pyro.sample("y_{}".format(t), dist.Normal(locs[x],scales[x]), obs=y)
            if (t < 4) and verbose:
                print("x_{}.shape = {}".format(t, x.shape))

    from pyro.optim import Adam
    from collections import defaultdict
    from tqdm.notebook import tqdm

    # We'll reuse the same guide and elbo.
    hmm_guide = AutoDiagonalNormal(poutine.block(hmm_model, 
                                                 expose=["transition", "locs","scales"]),
                                                    init_loc_fn=my_init_fn)

    pyro.clear_param_store()
    optim = pyro.optim.Adam({'lr': 0.05, 'betas': [0.8, 0.99]})
    elbo = TraceEnum_ELBO(max_plate_nesting=1)

    elbo.loss(hmm_model, hmm_guide, data, data_dim=data_dim);
    svi = SVI(hmm_model, hmm_guide, optim, loss=elbo)



    # Register hooks to monitor gradient norms.
    gradient_norms = defaultdict(list)
    for name, value in pyro.get_param_store().named_parameters():
        value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

    losses = []
    N_STEPS = 50
    for i in tqdm(range(N_STEPS)):
        loss = svi.step(data)
        losses.append(loss)


    map_estimates = hmm_guide(data)

    return map_estimates,losses,gradient_norms


# USE a viterbi algorithm to filter the data!
@infer_discrete(first_available_dim=-1, temperature=0)
@config_enumerate
def viterbi_decoder_z(data,transition,locs,scales):
    # takes the transition matrix and the inferred locations
    states = [0]
    for t in pyro.markov(range(len(data))):
        states.append(pyro.sample("states_{}".format(t),
                                  dist.Categorical(transition[states[-1]])))
        pyro.sample("obs_{}".format(t),
                    dist.Normal(locs[states[-1]], scales[states[-1]]),
                    obs=data[t])
    return states[1:]  # returns maximum likelihood states

def plot_pyro_loss(losses):
    from matplotlib import pyplot
#     %matplotlib inline
    pyplot.figure(figsize=(5,3), dpi=100).set_facecolor('white')
    pyplot.plot(losses,c='k')
    pyplot.xlabel('iters')
    pyplot.ylabel('loss')
    pyplot.yscale('log')
    pyplot.title('Convergence of SVI');
    pyplot.show()

def plot_pyro_gradients(gradient_norms):
    from matplotlib import pyplot
    
    pyplot.figure(figsize=(5,3), dpi=100).set_facecolor('white')
    for name, grad_norms in gradient_norms.items():
        pyplot.plot(grad_norms, label=name)
    pyplot.xlabel('iters')
    pyplot.ylabel('gradient norm')
    pyplot.yscale('log')
    pyplot.legend(loc='best')
    pyplot.title('Gradient norms during SVI');