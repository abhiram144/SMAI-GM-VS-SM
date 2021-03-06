#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:41:26 2020

@author: ljia

**This script demonstrates how to generate a graph preimage using Boria's method with cost matrices learning.**
"""

"""**1.   Get dataset.**"""

from gklearn.utils import Dataset, split_dataset_by_target

# Predefined dataset name, use dataset "MAO".
ds_name = 'MAO'
# The node/edge labels that will not be used in the computation.
irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']}

# Initialize a Dataset.
dataset_all = Dataset()
# Load predefined dataset "MAO".
dataset_all.load_predefined_dataset(ds_name)
# Remove irrelevant labels.
dataset_all.remove_labels(**irrelevant_labels)
# Split the whole dataset according to the classification targets.
datasets = split_dataset_by_target(dataset_all)
# Get the first class of graphs, whose median preimage will be computed.
dataset = datasets[0]
# dataset.cut_graphs(range(0, 10))
len(dataset.graphs)

"""**2.  Set parameters.**"""

import multiprocessing

# Parameters for MedianPreimageGenerator (our method).
mpg_options = {'fit_method': 'k-graphs', # how to fit edit costs. "k-graphs" means use all graphs in median set when fitting.
			   'init_ecc': [4, 4, 2, 1, 1, 1], # initial edit costs.
			   'ds_name': ds_name, # name of the dataset.
			   'parallel': True, # @todo: whether the parallel scheme is to be used.
			   'time_limit_in_sec': 0, # maximum time limit to compute the preimage. If set to 0 then no limit.
			   'max_itrs': 100, # maximum iteration limit to optimize edit costs. If set to 0 then no limit.
			   'max_itrs_without_update': 3, # If the times that edit costs is not update is more than this number, then the optimization stops.
			   'epsilon_residual': 0.01, # In optimization, the residual is only considered changed if the change is bigger than this number.
			   'epsilon_ec': 0.1, # In optimization, the edit costs are only considered changed if the changes are bigger than this number.
			   'verbose': 2 # whether to print out results.
               }
# Parameters for graph kernel computation.
kernel_options = {'name': 'PathUpToH', # use path kernel up to length h.
				  'depth': 9,
				  'k_func': 'MinMax',
				  'compute_method': 'trie',
				  'parallel': 'imap_unordered', # or None
				  'n_jobs': multiprocessing.cpu_count(),
				  'normalize': True, # whether to use normalized Gram matrix to optimize edit costs.
				  'verbose': 2 # whether to print out results.
                  }
# Parameters for GED computation.
ged_options = {'method': 'BIPARTITE', # use Bipartite huristic.
			   'initialization_method': 'RANDOM', # or 'NODE', etc.
			   'initial_solutions': 10, # when bigger than 1, then the method is considered mIPFP.
			   'edit_cost': 'CONSTANT', # use CONSTANT cost.
			   'attr_distance': 'euclidean', # the distance between non-symbolic node/edge labels is computed by euclidean distance.
			   'ratio_runs_from_initial_solutions': 1,
			   'threads': multiprocessing.cpu_count(), # parallel threads. Do not work if mpg_options['parallel'] = False.
			   'init_option': 'LAZY_WITHOUT_SHUFFLED_COPIES' # 'EAGER_WITHOUT_SHUFFLED_COPIES'
               }
# Parameters for MedianGraphEstimator (Boria's method).
mge_options = {'init_type': 'MEDOID', # how to initial median (compute set-median). "MEDOID" is to use the graph with smallest SOD.
			   'random_inits': 10, # number of random initialization when 'init_type' = 'RANDOM'.
			   'time_limit': 600, # maximum time limit to compute the generalized median. If set to 0 then no limit.
			   'verbose': 2, # whether to print out results.
			   'refine': False # whether to refine the final SODs or not.
               }
print('done.')

"""**3.   Run median preimage generator.**"""

from gklearn.preimage import MedianPreimageGeneratorPy

# Create median preimage generator instance.
mpg = MedianPreimageGeneratorPy()
# Add dataset.
mpg.dataset = dataset
# Set parameters.
mpg.set_options(**mpg_options.copy())
mpg.kernel_options = kernel_options.copy()
mpg.ged_options = ged_options.copy()
mpg.mge_options = mge_options.copy()
# Run.
mpg.run()

"""**4. Get results.**"""

# Get results.
import pprint
pp = pprint.PrettyPrinter(indent=4) # pretty print
results = mpg.get_results()
pp.pprint(results)

# Draw generated graphs.
def draw_graph(graph):
	import matplotlib.pyplot as plt
	import networkx as nx
	plt.figure()
	pos = nx.spring_layout(graph)
	nx.draw(graph, pos, node_size=500, labels=nx.get_node_attributes(graph, 'atom_symbol'), font_color='w', width=3, with_labels=True)
	plt.show()
	plt.clf()
	plt.close()
 
draw_graph(mpg.set_median)
draw_graph(mpg.gen_median)