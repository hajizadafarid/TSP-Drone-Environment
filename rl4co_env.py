import numpy as np
import os
import warnings
import collections
import copy
import time
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import Generator
from tensordict import TensorDict
import torch
from torch.distributions import Uniform

# Custom generator for the TSP-Drone problem
class TSPDroneGenerator(Generator):
    def __init__(
        self,
        num_loc: int = 20,  # Number of customer locations
        min_loc: float = 0.0,
        max_loc: float = 1.0,
    ):
        super().__init__()  # Initialize the base Generator class
        self.num_loc = num_loc  # Number of locations excluding depot
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.loc_sampler = torch.distributions.Uniform(
            low=min_loc, high=max_loc
        )

    def _generate(self, batch_size):
        batch_size = batch_size if isinstance(batch_size, int) else batch_size[0]  # Ensure batch_size is an integer
        truck_locs = self.loc_sampler.sample((batch_size, self.num_loc + 1, 2))

        # Sample demand: 1 for each customer location, 0 for depot
        demand = torch.cat((torch.ones(batch_size, self.num_loc, 1), torch.zeros(batch_size, 1, 1)), dim=1)
        
        # Combine locations and demand into a single tensor
        locs_with_demand = torch.cat((truck_locs, demand), dim=2)
        
        return TensorDict({"locs": locs_with_demand}, batch_size=[batch_size])

# Custom environment for the TSP-Drone problem, using RL4CO
class TSPDroneEnv(RL4COEnvBase):

    name = "tspd"

    def __init__(
        self,
        generator=TSPDroneGenerator,
        generator_params={},
        batch_size=1,
        v_t=1.0,
        v_d=1.0,
        input_data=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.generator = generator(**generator_params)
        self.v_t = v_t
        self.v_d = v_d
        self.input_data = input_data
        self.batch_size = torch.Size([batch_size])
        self.n_nodes = generator_params.get('num_loc', 20) + 1

    # Reset environment to the initial state
    def _reset(self, tensordict=None, **kwargs):
        if self.input_data is None:  # Only generate data if not provided
            self.input_data = self.generator(self.batch_size[0])["locs"]
        else:
            self.input_data = self.input_data.clone().detach()  

        self.batch_size = torch.Size([self.input_data.shape[0]])  # Ensure batch_size is set after input_data is initialized
        self.input_pnt = self.input_data[:, :, :2]  # Keep tensor for distance calculation
        self.dist_mat = torch.zeros([self.batch_size[0], self.n_nodes, self.n_nodes])
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                self.dist_mat[:, i, j] = torch.sqrt((self.input_pnt[:, i, 0] - self.input_pnt[:, j, 0]) ** 2 + (self.input_pnt[:, i, 1] - self.input_pnt[:, j, 1]) ** 2)
                self.dist_mat[:, j, i] = self.dist_mat[:, i, j]
       
        self.drone_mat = self.dist_mat / self.v_d
        avail_actions = torch.ones([self.batch_size[0], self.n_nodes, 2], dtype=torch.float32)
        avail_actions[:, self.n_nodes - 1, :] = torch.zeros([self.batch_size[0], 2])
        self.state = torch.ones([self.batch_size[0], self.n_nodes])
        self.state[:, self.n_nodes - 1] = torch.zeros([self.batch_size[0]])
        self.sortie = torch.zeros(self.batch_size[0])
        self.returned = torch.ones(self.batch_size[0])
        self.current_time = torch.zeros(self.batch_size[0])
        self.truck_loc = torch.ones([self.batch_size[0]], dtype=torch.int32) * (self.n_nodes - 1)
        self.drone_loc = torch.ones([self.batch_size[0]], dtype=torch.int32) * (self.n_nodes - 1)
        self.combined_nodes = torch.zeros([self.batch_size[0], self.n_nodes])
        self.combined_check = torch.zeros([self.batch_size[0], self.n_nodes])
        self.visit_times_truck = torch.zeros((self.batch_size[0], self.n_nodes))
        self.visit_times_drone = torch.zeros((self.batch_size[0], self.n_nodes))
        dynamic = torch.zeros([self.batch_size[0], self.n_nodes, 2], dtype=torch.float32)
        dynamic[:, :, 0] = self.dist_mat[:, self.n_nodes - 1]
        dynamic[:, :, 1] = self.drone_mat[:, self.n_nodes - 1]

        tensordict_out = TensorDict({
            "dynamic": dynamic,
            "avail_actions": avail_actions
        }, batch_size=[self.batch_size[0]])

        return tensordict_out

    # Step function to simulate environment dynamics
    def _step(self, tensordict: TensorDict):
        idx_truck = tensordict.get("idx_truck")
        idx_drone = tensordict.get("idx_drone")
        time_vec_truck = tensordict.get("time_vec_truck")
        time_vec_drone = tensordict.get("time_vec_drone")
        terminated = tensordict.get("terminated")
        
        old_sortie = self.sortie.clone()
        
        t_truck = self.dist_mat[torch.arange(self.batch_size[0]), self.truck_loc, idx_truck]
        t_drone = self.drone_mat[torch.arange(self.batch_size[0]), self.drone_loc, idx_drone]
        
        A = t_truck + torch.eq(t_truck, torch.zeros(self.batch_size[0])).float() * torch.ones(self.batch_size[0]) * 10000
        B = t_drone + torch.eq(t_drone, torch.zeros(self.batch_size[0])).float() * torch.ones(self.batch_size[0]) * 10000
        C = time_vec_truck[:, 1] + torch.eq(time_vec_truck[:, 1], torch.zeros(self.batch_size[0])).float() * torch.ones(self.batch_size[0]) * 10000
        D = time_vec_drone[:, 1] + torch.eq(time_vec_drone[:, 1], torch.zeros(self.batch_size[0])).float() * torch.ones(self.batch_size[0]) * 10000
        time_step = torch.min(torch.stack([A, B, C, D]), dim=0)[0]
        
        b_s = torch.where(terminated == 1)[0]
        time_step[b_s] = torch.zeros(len(b_s), dtype=time_step.dtype)
        self.time_step = time_step
        self.current_time += time_step
        
        time_vec_truck[:, 1] += torch.logical_and(torch.eq(time_vec_truck[:, 1], torch.zeros(self.batch_size[0])), torch.gt(t_truck, torch.zeros(self.batch_size[0]))).float() * (t_truck - time_step) - torch.gt(time_vec_truck[:, 1], torch.zeros(self.batch_size[0])) * time_step
        
        time_vec_drone[:, 1] += torch.logical_and(torch.eq(time_vec_drone[:, 1], torch.zeros(self.batch_size[0])), 
                                            torch.gt(t_drone, torch.zeros(self.batch_size[0]))).float() * (t_drone - time_step) - \
                                torch.gt(time_vec_drone[:, 1], torch.zeros(self.batch_size[0])) * time_step
        
        self.truck_loc += torch.eq(time_vec_truck[:, 1], torch.zeros(self.batch_size[0])).long() * (idx_truck - self.truck_loc)
        self.drone_loc += torch.eq(time_vec_drone[:, 1], torch.zeros(self.batch_size[0])).long() * (idx_drone - self.drone_loc)
        
        time_vec_truck[:, 0] = torch.logical_and(torch.lt(time_step, t_truck), torch.gt(time_vec_truck[:, 1], torch.zeros(self.batch_size[0]))).float() * idx_truck
        time_vec_drone[:, 0] = torch.logical_and(torch.lt(time_step, t_drone), torch.gt(time_vec_drone[:, 1], torch.zeros(self.batch_size[0]))).float() * idx_drone

        self.visit_times_truck[torch.arange(self.batch_size[0]), idx_truck] += torch.eq(time_vec_truck[:, 1], 0).float()
        self.visit_times_drone[torch.arange(self.batch_size[0]), idx_drone] += torch.eq(time_vec_drone[:, 1], 0).float()

        b_s = torch.where(torch.eq(time_vec_truck[:, 1], torch.zeros(self.batch_size[0])))[0]
        self.state[b_s, idx_truck[b_s]] = torch.zeros(len(b_s))
        idx_satis = torch.where(torch.lt(self.sortie.float() - torch.eq(time_vec_drone[:, 1], 0).float(), torch.zeros(self.batch_size[0])))[0]
        self.state[idx_satis, idx_drone[idx_satis]] -= torch.eq(time_vec_drone[idx_satis, 1], torch.zeros(len(idx_satis))).float() * self.state[idx_satis, idx_drone[idx_satis]]
        self.sortie[idx_satis] = torch.ones(len(idx_satis))
       
        a = torch.eq((self.truck_loc == self.drone_loc).float() + (time_vec_drone[:, 1] == 0).float() + (time_vec_truck[:, 1] == 0).float(), 3)
        b = torch.eq((self.combined_nodes[torch.arange(self.batch_size[0]), self.truck_loc] == 1).float() + a.float(), 2)
        idx_stais = torch.where(torch.unsqueeze(a, 1))[0]
        self.sortie[idx_stais] = torch.zeros(len(idx_stais))
        
        self.returned = torch.ones(self.batch_size[0], dtype=torch.float) - torch.eq((old_sortie == 1).float() + (self.sortie == 1).float() + (time_vec_drone[:, 1] == 0).float(), 3).float()
        self.returned[idx_stais] = torch.ones(len(idx_stais), dtype=torch.float)
       
        self.combined_nodes[idx_stais, self.truck_loc[idx_stais]] = 1
        b_s = torch.where(b)[0]
        self.combined_check[b_s, idx_truck[b_s]] = 1

        avail_actions = torch.zeros([self.batch_size[0], self.n_nodes, 2], dtype=torch.float32)
        
        b_s = torch.where(torch.unsqueeze(time_vec_truck[:, 1], 1) > 0)[0]
        idx_fixed = time_vec_truck[b_s, torch.zeros(len(b_s), dtype=torch.int64)]
        avail_actions[b_s, idx_fixed.long(), 0] = torch.ones(len(b_s))
        
        b_s_d = torch.where(torch.unsqueeze(time_vec_drone[:, 1], 1) > 0)[0]
        idx_fixed_d = time_vec_drone[b_s_d, torch.zeros(len(b_s_d), dtype=torch.int64)]
        avail_actions[b_s_d, idx_fixed_d.long(), 1] = torch.ones(len(b_s_d))
       
        a = torch.eq(torch.ge(time_vec_truck[:, 1], 0).float() + torch.eq(time_vec_drone[:, 1], 0).float(), 2)
        b_s = torch.where(torch.unsqueeze(a, 1))[0]
        avail_actions[b_s, :, 1] = torch.gt(self.state[b_s, :], 0).float()
        
        a = torch.eq(torch.eq(self.returned, 0).float() + torch.eq(time_vec_drone[:, 1], 0).float(), 2)
        b_s = torch.where(torch.unsqueeze(a, 1))[0]
        avail_actions[b_s, :, 1] = 0
        avail_actions[b_s, self.drone_loc[b_s], 1] = 1
        
        a = torch.eq(torch.eq(time_vec_drone[:, 1], 0).float() + torch.eq(self.sortie, 1).float() + torch.eq(self.returned, 1).float(), 3)
        b_s = torch.where(a)[0]
        avail_actions[b_s, :, 1] += (self.combined_nodes[b_s] - self.combined_check[b_s]) * torch.unsqueeze(torch.gt(self.state[b_s].sum(dim=1), 2).float(), 1) * torch.ones([len(b_s), self.n_nodes])
        avail_actions[b_s, :, 1] = torch.gt(avail_actions[b_s, :, 1], 0).float()

        b_s = torch.where(torch.unsqueeze(time_vec_truck[:, 1], 1) == 0)[0]
        avail_actions[b_s, :, 0] += torch.gt(self.state[b_s, :], 0)
        avail_actions[b_s, :, 0] += (self.combined_nodes[b_s] - self.combined_check[b_s]) * torch.unsqueeze(torch.gt(self.state[b_s].sum(dim=1), 2).float(), 1) * torch.ones([len(b_s), self.n_nodes])

        a = torch.eq(torch.eq(self.sortie, 0).float() + torch.gt(time_vec_drone[:, 1], 0).float() + torch.eq(time_vec_truck[:, 1], 0).float(), 3)
        b_s_s = torch.where(torch.unsqueeze(a, 1))[0]
        idx_fixed_d = time_vec_drone[b_s_s, torch.zeros(len(b_s_s), dtype=torch.int64)]
        avail_actions[b_s_s, idx_fixed_d.long(), 0] = 0

        a = torch.eq(torch.eq(self.truck_loc, time_vec_drone[:, 0]).float() + torch.gt(time_vec_drone[:, 1], 0).float() + torch.eq(time_vec_truck[:, 1], 0).float(), 3)
        b_s = torch.where(torch.unsqueeze(a, 1))[0]
        avail_actions[b_s, self.truck_loc[b_s], 0] = 1

        a = torch.eq(torch.eq(self.returned, 0).float() + torch.eq(time_vec_drone[:, 1], 0).float() + torch.eq(time_vec_truck[:, 1], 0).float(), 3)
        b_s = torch.where(torch.unsqueeze(a, 1))[0]
        avail_actions[b_s, self.drone_loc[b_s], 0] = 1

        a = torch.eq(torch.eq(self.state.sum(dim=1), 1).float() + torch.eq((avail_actions[:, :, 0] == avail_actions[:, :, 1]).sum(dim=1), self.n_nodes).float() + torch.eq(self.drone_loc, self.truck_loc).float(), 3)
        b_s = torch.where(a)[0]
        avail_actions[b_s, :, 0] = torch.zeros(self.n_nodes)

        a = torch.eq(torch.eq(self.state.sum(dim=1), 1).float() + torch.eq(time_vec_drone[:, 1], 0).float() + torch.gt(time_vec_truck[:, 1], 0).float() + torch.eq(self.returned, 1).float(), 4)
        b_s = torch.where(a)[0]
        avail_actions[b_s, :, 1] = torch.zeros(self.n_nodes)

        avail_actions[:, self.n_nodes-1, 0] += torch.eq(avail_actions[:, :, 0].sum(dim=1), 0)
        avail_actions[:, self.n_nodes-1, 1] += torch.eq(avail_actions[:, :, 1].sum(dim=1), 0)

        dynamic = torch.zeros([self.batch_size[0], self.n_nodes, 2], dtype=torch.float32)
        dynamic[:, :, 0] = self.dist_mat[torch.arange(self.batch_size[0]), self.truck_loc]
        dynamic[:, :, 1] = self.drone_mat[torch.arange(self.batch_size[0]), self.drone_loc]

        terminated = torch.logical_and(torch.eq(self.truck_loc, self.n_nodes-1), torch.eq(self.drone_loc, self.n_nodes-1)).float()
        tensordict_out = TensorDict({
                "dynamic": dynamic,
                "avail_actions": avail_actions,
                "terminated": terminated,
                "time_vec_truck": time_vec_truck,
                "time_vec_drone": time_vec_drone
            }, batch_size=[self.batch_size[0]])

        return tensordict_out

    def _calculate_cost(self):
        return torch.max(self.current_time)

    _get_reward = _calculate_cost
