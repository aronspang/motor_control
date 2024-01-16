import numpy as np
import matplotlib.pyplot as plt
import os

# set to actual value
eval_episode = 50

# Define the log directory
log_dir = os.path.join(os.getcwd(), "logs")

# List all files in the log directory
all_files = os.listdir(log_dir)

# Filter files that contain "dense" or "sparse"
dense_files = [f for f in all_files if "dense" in f]
sparse_files = [f for f in all_files if "sparse" in f]

# Function to calculate mean successes
def calculate_mean_successes(file_list):
    mean_successes_list = []
    timesteps_list = []
    for file in file_list:
        data = np.load(os.path.join(log_dir, file))
        mean_successes = np.mean(data['successes'], axis=1)
        mean_successes_list.append(mean_successes)
        timesteps_list.append(data['timesteps'])
    
    # Concatenate all mean successes and timesteps
    all_mean_successes = np.mean(mean_successes_list, axis=0)
    all_timesteps = np.mean(timesteps_list, axis=0)
    return all_timesteps, all_mean_successes

# Calculate mean successes for dense and sparse data
dense_timesteps, dense_mean_successes = calculate_mean_successes(dense_files)
sparse_timesteps, sparse_mean_successes = calculate_mean_successes(sparse_files)

# Plotting both graphs in one figure, one above the other
fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # Two rows, one column

# Plot for sparse data
axs[0].scatter(sparse_timesteps, sparse_mean_successes, alpha=1, label='Sparse Data Success Rate', color='teal', marker='x')
axs[0].set_title(f"Sparse Rewards Success Rate (mean of n={len(dense_files)} models)")
axs[0].set_ylabel(f'Mean success rate (n={eval_episode} eval episodes)')
axs[0].grid(True)
axs[0].legend()


# Plot for dense data
axs[1].scatter(dense_timesteps, dense_mean_successes, alpha=1, label='Dense Data Success Rate', color='teal', marker='x')
axs[1].set_title(f"Dense Rewards Success Rate (mean of n={len(dense_files)} models)")
axs[1].set_xlabel('Number of training timesteps after which evaluation is run')
axs[1].set_ylabel(f'Mean success rate (n={eval_episode} eval episodes)')
axs[1].grid(True)
axs[1].legend()

plt.show()
