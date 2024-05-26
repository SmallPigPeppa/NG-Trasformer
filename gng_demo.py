from neupy import algorithms
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


data, _ = make_moons(10000, noise=0.06, random_state=0)
plt.scatter(*data.T)
plt.show()

from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from neupy import algorithms, utils

utils.reproducible()

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
plt.scatter(*data.T, alpha=0.02)
plt.xticks([], [])
plt.yticks([], [])


def animate(i):
    for line in animate.prev_lines:
        line.remove()

    # Training will slow down overtime and we increase number
    # of data samples for training
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
HTML(anim.to_html5_video())