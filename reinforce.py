import os
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import Tensor
from policies import NeuralNetwork, LinearRegression
from utilities import pretraining, compute_run

np.random.seed(seed=666)
torch.manual_seed(666)

# specifications for dynamic system
ti = 0  # define initial time
tf = 1  # define final time
divisions = 20
dtime = (tf-ti)/divisions
params = {'a': 0.5, 'b': 1}

# define policy network and other learning/reporting parameters
hidden_layers_size = 15
input_size = 3
output_size = 1
# policy = NeuralNetwork(hidden_layers_size, input_size, output_size)
policy = LinearRegression(input_size, output_size)

# Pre-training
initial_objective = torch.tensor([i_PT * 5.0 / divisions for i_PT in range(divisions)])
initial_state = np.array([1, 0])
state_range_PT, control_range_PT = pretraining(
    policy, initial_objective, params, initial_state, divisions, ti, tf, dtime,
    learning_rate=1e-1, epochs=100, pert_size=0.1
    )

y1_PT = [state[0] for state in state_range_PT]
y2_PT = [state[1] for state in state_range_PT]
pretrained_policy_control = [None for i_PT in range(divisions)]

# evaluate model with real data
for point in range(divisions):
    state_PT = state_range_PT[point]
    output = policy(state_PT)
    output_np = output.data.numpy()
    pretrained_policy_control[point] = output_np[0]

label_size = 10
grid_shape = (3, 1) # (rows, columns)
fig = plt.figure(0)

time_array = [ti + div * dtime for div in range(divisions)]

plt.subplot2grid(grid_shape, (0, 0))
plt.plot(time_array, y1_PT)
plt.grid(axis='x') # , ls='--', lw=.5, c='k', alpha=.3
plt.ylabel('y1', fontsize=label_size)
plt.xlabel('time', fontsize=label_size)

plt.subplot2grid(grid_shape, (1, 0))
plt.plot(time_array, y2_PT)
plt.grid(axis='x')
plt.ylabel('y2', fontsize=label_size)
plt.xlabel('time', fontsize=label_size)

plt.subplot2grid(grid_shape, (2, 0))
plt.plot(time_array, pretrained_policy_control)
plt.plot(time_array, initial_objective.numpy(), 'ro')
plt.grid(axis='x')
plt.ylabel('u', fontsize=label_size)
plt.xlabel('time', fontsize=label_size)

# fig.suptitle("Hola")
plt.show()

# ---------------------------------------------------
#                  MAIN LOOP start
# ---------------------------------------------------

# problem parameters
episode_n = 100000
record_n = 1000
std_sqr_red = (1, record_n)

std_sqr = 1.0  # remember this is reduced first iter !!
episode_update_n = 1  # every how many episodes I update !!
rewards_l = [None for i in range(episode_update_n)]  # keep track of rewards per episode
log_probs_l = [[None for j in range(divisions)] for i in range(episode_update_n)]
epi_n = -1  # because I sum one
rewards_record = []

optimizer = optim.Adam(policy.parameters(), lr=1e-4)

# create folders to save pictures and model
os.makedirs('figures', exist_ok=True)
os.makedirs('serializations', exist_ok=True)
for i_episode in range(episode_n):
    # COMPUTE EPISODE FOR LATER REINFORCEMENT LEARNING
    if i_episode % std_sqr_red[1] == 0:  # diminish std to focus more on explotation
        std_sqr = std_sqr * std_sqr_red[0]
    # run episode with network policy
    initial = np.array([1, 0])
    rewards_l[epi_n] = compute_run(
        policy, initial, params, log_probs_l,
        dtime, divisions, ti, tf, std_sqr, epi_n,
        plot=False
        )
    epi_n = epi_n + 1
    # TRAIN CURRENT POLICY NETWORK USING COMPUTED TRAJECTORIES
    if i_episode != 0 and i_episode % episode_update_n == 0:
        gamma = 0.99
        loss = 0
        for i_ep in range(len(log_probs_l)):
            R = torch.zeros(1)  # , 1)
            for i_j in reversed(range(len(log_probs_l[i_ep]))):  # not sure why
                R = gamma * R + rewards_l[i_ep]
                loss = loss - (
                    log_probs_l[i_ep][i_j] * R.expand_as(log_probs_l[i_ep][i_j])
                    ).sum()
                # A3C: We add entropy to the loss to encourage exploration
                # https://github.com/dennybritz/reinforcement-learning/issues/34
        loss = loss / (episode_update_n)
        optimizer.zero_grad()
        loss.backward()
        # utils.clip_grad_norm(policy.parameters(), 40)
        # https://discuss.pytorch.org/t/about-torch-nn-utils-clip-grad-norm/13873/2
        optimizer.step()
        # define new rewards and log probs for next episodes to train (like a "zero_grad")
        epi_n = 0
        rewards_l = [None for i in range(episode_update_n)]
        log_probs_l = [[None for j in range(divisions)] for i in range(episode_update_n)]
    # PLOT CURRENT MEAN POLICY
    if i_episode % record_n == 0:
        r_report, y1_l, y2_l, t_l, U_l = compute_run(
            policy, np.array([1, 0]), params, log_probs_l,
            dtime, divisions, ti, tf, std_sqr, epi_n,
            plot=True
            )

        print('i_episode = ', i_episode, '     current_reward = ', r_report)
        rewards_record.append(r_report)
        print('std_sqr = ', std_sqr)
        plt.subplot2grid((2, 4), (0, 0), colspan=2)
        plt.plot(y1_l)
        plt.ylabel('y1 ', rotation=360, fontsize=15)
        plt.xlabel('time', fontsize=15)

        plt.subplot2grid((2, 4), (0, 2), colspan=2)
        plt.plot(y2_l)
        plt.ylabel('y2 ', rotation=360, fontsize=15)
        plt.xlabel('time', fontsize=15)

        plt.subplot2grid((2, 4), (1, 0), colspan=2)
        plt.plot(U_l)
        plt.ylabel('u', rotation=360, fontsize=15)
        plt.xlabel('time', fontsize=15)
        plt.savefig(
            join('figures', 'profile_iter_a_' + str(i_episode) + '_REINFORCE_v3.png')
            )
        plt.close()
        torch.save(policy, join('serializations', 'ann_pretrain_10-6.pt'))

print('finished learning')
print('std_sqr = ', std_sqr)
plt.plot(rewards_record)
plt.ylabel('reward value')
plt.show()
