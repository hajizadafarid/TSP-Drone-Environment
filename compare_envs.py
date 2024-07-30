import numpy as np
from original_env import OriginalEnv, create_test_dataset
from rl4co_env import TSPDroneEnv, TSPDroneGenerator
import torch
import matplotlib.pyplot as plt
import os
from tensordict import TensorDict

args = {
    'random_seed': 16,
    'test_size': 5,
    'n_nodes': 20,
    'data_dir': './data',
    'v_t': 2.0,
    'v_d': 1.0,
    'batch_size': 5
}

# Create test data using the original environment's data generator function
test_data, _ = create_test_dataset(args)

# Convert test data to tensor format compatible with RL4CO environment
test_data_tensor = torch.tensor(test_data[:, :, :2], dtype=torch.float32)

# Initialize both environments with the same data
original_env = OriginalEnv(args, test_data)
rl4co_env = TSPDroneEnv(generator_params={'num_loc': args['n_nodes'] - 1, 'min_loc': 0.0, 'max_loc': 1.0}, batch_size=args['batch_size'], v_t=args['v_t'], v_d=args['v_d'], input_data=test_data_tensor)

# Reset both environments
original_dynamic, original_avail_actions = original_env.reset()
rl4co_tensordict = rl4co_env.reset()
rl4co_dynamic = rl4co_tensordict["dynamic"]
rl4co_avail_actions = rl4co_tensordict["avail_actions"]

# Visualize the initial locations
def plot_initial_locations(data, title):
    for i in range(data.shape[0]):
        plt.scatter(data[i, :-1, 0], data[i, :-1, 1], c='blue', label='Customers')
        plt.scatter(data[i, -1, 0], data[i, -1, 1], c='red', label='Depot')
        plt.title(f"{title} - Batch__ {i}")
        plt.legend()
        plt.savefig(os.path.join(args['data_dir'], f"{title}_Batch__{i}.png"))
        plt.show()

# Plot initial locations for both environments
plot_initial_locations(test_data[:, :, :2], "Original Environment Initial Locations")
plot_initial_locations(test_data_tensor.numpy(), "RL4CO Environment Initial Locations")

# Compare the initial states, reset function
assert np.allclose(original_dynamic, rl4co_dynamic.numpy(), atol=1e-5), "Initial dynamic states do not match!"
assert np.allclose(original_avail_actions, rl4co_avail_actions.numpy(), atol=1e-5), "Initial available actions do not match!"

num_steps = 10
truck_paths_orig = [[] for _ in range(args['batch_size'])]
drone_paths_orig = [[] for _ in range(args['batch_size'])]
truck_paths_rl4co = [[] for _ in range(args['batch_size'])]
drone_paths_rl4co = [[] for _ in range(args['batch_size'])]

for step in range(num_steps):
    idx_truck = np.random.randint(0, args['n_nodes'], size=args['batch_size'])
    idx_drone = np.random.randint(0, args['n_nodes'], size=args['batch_size'])
    time_vec_truck = np.zeros((args['batch_size'], 2))
    time_vec_drone = np.zeros((args['batch_size'], 2))
    terminated = np.zeros(args['batch_size'], dtype=bool)
    
    time_vec_truck_orig = time_vec_truck.copy()
    time_vec_drone_orig = time_vec_drone.copy()

    # Step in original environment
    original_dynamic, original_avail_actions, original_terminated, original_time_vec_truck, original_time_vec_drone = original_env.step(idx_truck, idx_drone, time_vec_truck_orig, time_vec_drone_orig, terminated)
    
    # Step in RL4CO environment
    idx_truck = torch.from_numpy(idx_truck)
    idx_drone = torch.from_numpy(idx_drone)
    time_vec_truck_rl4co = torch.from_numpy(time_vec_truck)
    time_vec_drone_rl4co = torch.from_numpy(time_vec_drone)
    terminated = torch.from_numpy(terminated)

    tensordict_in = TensorDict({
        "idx_truck": idx_truck,
        "idx_drone": idx_drone,
        "time_vec_truck": time_vec_truck_rl4co,
        "time_vec_drone": time_vec_drone_rl4co,
        "terminated": terminated
    }, batch_size=[args['batch_size']])

    rl4co_tensordict_out = rl4co_env.step(tensordict_in)
    rl4co_dynamic = rl4co_tensordict_out["next"]["dynamic"]
    rl4co_avail_actions = rl4co_tensordict_out["next"]["avail_actions"]

    # Append paths
    for i in range(args['batch_size']):
        truck_paths_orig[i].append(test_data[i, original_env.truck_loc[i], :2])
        drone_paths_orig[i].append(test_data[i, original_env.drone_loc[i], :2])
        truck_paths_rl4co[i].append(test_data_tensor[i, rl4co_env.truck_loc[i], :2].numpy())
        drone_paths_rl4co[i].append(test_data_tensor[i, rl4co_env.drone_loc[i], :2].numpy())
    
    print(f"Step {step + 1}:")
    assert np.allclose(original_dynamic, rl4co_dynamic.numpy(), atol=1e-5), f"Dynamic states do not match at step {step}!"
    assert np.allclose(original_avail_actions, rl4co_avail_actions.numpy(), atol=1e-5), f"Available actions do not match at step {step}!"

# Convert paths to numpy arrays
truck_paths_orig = [np.array(path) for path in truck_paths_orig]
drone_paths_orig = [np.array(path) for path in drone_paths_orig]
truck_paths_rl4co = [np.array(path) for path in truck_paths_rl4co]
drone_paths_rl4co = [np.array(path) for path in drone_paths_rl4co]

# Plot paths
def plot_paths(paths, data, title):
    for i in range(len(paths)):
        plt.scatter(data[i, :-1, 0], data[i, :-1, 1], c='blue', label='Customers')
        plt.scatter(data[i, -1, 0], data[i, -1, 1], c='red', label='Depot')
        truck_path = np.array(paths[i][0])
        drone_path = np.array(paths[i][1])
        plt.plot(truck_path[:, 0], truck_path[:, 1], c='green', label='Truck Path')
        plt.plot(drone_path[:, 0], drone_path[:, 1], c='purple', label='Drone Path')
        plt.title(f"{title} - Batch__ {i}")
        plt.legend()
        plt.savefig(os.path.join(args['data_dir'], f"{title}_Batch__{i}.png"))
        plt.show()

plot_paths(list(zip(truck_paths_orig, drone_paths_orig)), test_data[:, :, :2], "Original Environment Paths")
plot_paths(list(zip(truck_paths_rl4co, drone_paths_rl4co)), test_data_tensor.numpy(), "RL4CO Environment Paths")

# Compare final rewards
original_reward = original_env.calculate_cost()
rl4co_reward = rl4co_env._calculate_cost()

assert np.isclose(original_reward, rl4co_reward.item(), atol=1e-5), "Final rewards do not match!"

print("Both environments match in behavior and reward!")
