from neupy import algorithms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os
from tqdm import tqdm
from IPython.display import HTML

# Create result directory if it doesn't exist
result_dir = 'result'
os.makedirs(result_dir, exist_ok=True)


def generate_data(distribution_type, n_samples=10000, n_clusters=4, cluster_std=0.5, random_state=0):
    np.random.seed(random_state)
    data = []
    labels = []
    centers = []

    def is_overlapping(new_center, centers, min_dist=1.5):
        if not centers:
            return False
        distances = np.linalg.norm(np.array(centers) - new_center, axis=1)
        return np.any(distances < min_dist)

    for i in range(n_clusters):
        while True:
            center = np.random.uniform(-5, 5, 2)
            if not is_overlapping(center, centers):
                centers.append(center)
                break

        if distribution_type == 'circles':
            cluster_data = np.random.normal(loc=center, scale=cluster_std, size=(n_samples // n_clusters, 2))
        elif distribution_type == 'squares':
            size = 4  # Increase the size of the squares
            cluster_data = np.random.uniform(low=center - size / 2, high=center + size / 2,
                                             size=(n_samples // n_clusters, 2))
        else:
            raise ValueError("Unsupported distribution type")
        data.append(cluster_data)
        labels.append(np.full((n_samples // n_clusters,), i))

    data = np.vstack(data)
    labels = np.concatenate(labels)
    return data, labels


# Set the distribution type here
distribution_type = 'squares'  # Options: 'circles', 'squares'
distribution_type = 'circles'  # Options: 'circles', 'squares'

# Generate dataset
data, labels = generate_data(distribution_type)
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
# plt.title(f'{distribution_type} distribution with {len(np.unique(labels))} clusters')
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

fig = plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.2)
plt.xticks([], [])
plt.yticks([], [])


def animate(i):
    for line in animate.prev_lines:
        line.remove()

    # Training will slow down overtime and we increase number of data samples for training
    n = int(0.5 * gng.n_iter_before_neuron_added * (1 + i // 100))
    sampled_data_ids = np.random.choice(len(data), n)
    sampled_data = data[sampled_data_ids, :]
    gng.train(sampled_data, epochs=1)

    lines = []
    for node_1, node_2 in gng.graph.edges:
        weights = np.concatenate([node_1.weight, node_2.weight])
        line, = plt.plot(*weights.T, color='black')
        plt.setp(line, linewidth=1, color='black')
        lines.append(line)
        lines.append(plt.scatter(*weights.T, color='black', s=10))

    animate.prev_lines = lines
    return lines


animate.prev_lines = []
anim = animation.FuncAnimation(fig, animate, tqdm(np.arange(220)), interval=30, blit=True)

# Save the animation
anim.save(os.path.join(result_dir, 'gng_animation.mp4'), writer='ffmpeg', fps=30)

# Display the animation as HTML
HTML(anim.to_html5_video())
