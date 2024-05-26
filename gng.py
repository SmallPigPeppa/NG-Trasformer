from neupy import algorithms
from sklearn.datasets import make_blobs

data, _ = make_blobs(
    n_samples=1000,
    n_features=2,
    centers=2,
    cluster_std=0.4,
)

neural_gas = algorithms.GrowingNeuralGas(
    n_inputs=2,
    shuffle_data=True,
    verbose=True,
    max_edge_age=10,
    n_iter_before_neuron_added=50,
    max_nodes=100,
)

print(neural_gas.graph.n_nodes)

len(neural_gas.graph.edges)

edges = list(neural_gas.graph.edges.keys())
neuron_1, neuron_2 = edges[0]

print(neuron_1.weight)

print(neuron_2.weight)