from neupy import algorithms
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import os
from tqdm import tqdm
import numpy as np

# Create result directory if it doesn't exist
result_dir = 'result'
os.makedirs(result_dir, exist_ok=True)

# Generate moon dataset
data, _ = make_moons(1000, noise=0.06, random_state=0)
plt.scatter(*data.T)
plt.show()

# Initialize Growing Neural Gas algorithm
gng = algorithms.GrowingNeuralGas(
    n_inputs=2,
    n_start_nodes=2,
    shuffle_data=True,
    verbose=False,
    step=0.1,
    neighbour_step=0.001,
    max_edge_age=50,
    max_nodes=100,
    n_iter_before_neuron_added=100,
    after_split_error_decay_rate=0.5,
    error_decay_rate=0.995,
    min_distance_for_update=0.2,
)

# Plotting and saving each step
fig, ax = plt.subplots()
ax.scatter(*data.T, alpha=0.02)
ax.set_xticks([])
ax.set_yticks([])

for i in tqdm(range(220)):
    # Training will slow down overtime and we increase number of data samples for training
    n = int(0.5 * gng.n_iter_before_neuron_added * (1 + i // 100))
    sampled_data_ids = np.random.choice(len(data), n)
    sampled_data = data[sampled_data_ids, :]
    gng.train(sampled_data, epochs=1)

    ax.clear()
    ax.scatter(*data.T, alpha=0.02)
    for node_1, node_2 in gng.graph.edges:
        weights = np.concatenate([node_1.weight, node_2.weight])
        ax.plot(*weights.T, color='black', linewidth=1)
        ax.scatter(*weights.T, color='black', s=10)

    ax.set_xticks([])
    ax.set_yticks([])

    # Save the current plot
    fig.savefig(os.path.join(result_dir, f'step_{i:03d}.png'))

plt.close(fig)
