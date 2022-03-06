import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import time
import random
import math
import copy
from matplotlib import pyplot as plt
from ofa.model_zoo import ofa_net
from ofa.utils import download_url
from ofa.tutorial import AccuracyPredictor, FLOPsTable, LatencyTable, EvolutionFinder
from ofa.tutorial import evaluate_ofa_subnet, evaluate_ofa_specialized

# Set random seed to get same result
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!' % random_seed)

# Check if CUDA is available
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')

# Import pretrained super network
ofa_network = ofa_net('ofa_proxyless_d234_e346_k357_w1.3', pretrained=True)
print('The OFA Network is ready.')

# Note 10 accuracy predictor & Note 10 latency lookup table.
accuracy_predictor = AccuracyPredictor(pretrained=True, device='cuda:0' if cuda_available else 'cpu')
print('The accuracy predictor is ready!')
target_hardware = 'note10'
latency_table = LatencyTable(device=target_hardware)
print('The Latency lookup table on %s is ready!' % target_hardware)
print(accuracy_predictor.model)

# FLOPs lookup table.
flops_lookup_table = FLOPsTable(device='cuda:0' if cuda_available else 'cpu', batch_size=1, )
print('The FLOPs lookup table is ready!')

# Hyper-parameters for the evolutionary search process. It will show the predicted final ImageNet accuracy of the
# search sub-net.
latency_constraint = 25  # ms, suggested range [15, 33] ms
P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': target_hardware,  # Let's do FLOPs-constrained search
    'efficiency_constraint': latency_constraint,
    'mutate_prob': 0.1,  # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5,  # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': latency_table,  # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor,  # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
}

# Build the evolution finder
finder = EvolutionFinder(**params)

# Start searching
result_lis = []
st = time.time()
best_valids, best_info = finder.run_evolution_search()
result_lis.append(best_info)
ed = time.time()
print('Found best architecture on %s with latency <= %.2f ms in %.2f seconds! '
      'It achieves %.2f%s predicted accuracy with %.2f ms latency on %s.' %
      (target_hardware, latency_constraint, ed - st, best_info[0] * 100, '%', best_info[-1], target_hardware))

# Visualize the architecture of the searched sub-net
_, net_config, latency = best_info
ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
print('Architecture of the searched sub-net:')
print(ofa_network.module_str)

