import numpy as np
import matplotlib.pyplot as plt
import os

# Define a callback for evaluation
log_dir = os.path.join(os.getcwd(), "logs")

# Load the npz files
dense_data_path = os.path.join(log_dir, 'dense_evaluations_name_20240103164533.npz')
sparse_data_path = os.path.join(log_dir, 'sparse_evaluations_name_20240103180904.npz')

# Load the data:
dense_data = np.load(dense_data_path)
sparse_data = np.load(sparse_data_path)

# Checking the shape of 'timesteps' and 'successes' for both dense and sparse data
dense_timesteps_shape = dense_data['timesteps'].shape
dense_successes_shape = dense_data['successes'].shape
sparse_timesteps_shape = sparse_data['timesteps'].shape
sparse_successes_shape = sparse_data['successes'].shape

print(dense_data['successes'][0])

# Calculate the mean success rate for each timestep
dense_mean_successes = np.mean(dense_data['successes'], axis=1)
sparse_mean_successes = np.mean(sparse_data['successes'], axis=1)

# Plotting both graphs in one figure, one above the other
fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # Two rows, one column

# Plot for dense data
axs[0].scatter(dense_data['timesteps'], dense_mean_successes, alpha=0.7, label='Dense Data Success Rate')
axs[0].set_title("Dense Data Success Rate per Timestep")
axs[0].set_xlabel('Timesteps')
axs[0].set_ylabel('Mean Success Rate')
axs[0].grid(True)
axs[0].legend()

# Plot for sparse data
axs[1].scatter(sparse_data['timesteps'], sparse_mean_successes, alpha=0.7, label='Sparse Data Success Rate')
axs[1].set_title("Sparse Data Success Rate per Timestep")
axs[1].set_xlabel('Timesteps')
axs[1].set_ylabel('Mean Success Rate')
axs[1].grid(True)
axs[1].legend()


plt.show()
