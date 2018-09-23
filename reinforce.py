import os
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import Tensor
from policies import NeuralNetwork, LinearRegression
from utilities import pretraining, compute_run
from plots import plot_state_policy_evol

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
policy = NeuralNetwork(hidden_layers_size, input_size, output_size)
# policy = LinearRegression(input_size, output_size)

# pretrain policy with linear policy
pretraining_objective = [div * 5.0 / divisions for div in range(divisions)]
initial_state = np.array([1, 0])
state_range_PT, control_range_PT = pretraining(
    policy, pretraining_objective, params, initial_state, divisions, ti, tf, dtime,
    learning_rate=1e-1, epochs=200, pert_size=0.0
    )

y1_PT = [state[0] for state in state_range_PT]
y2_PT = [state[1] for state in state_range_PT]
pretrained_policy_control = [None for div in range(divisions)]

# evaluate model with real data
for point in range(divisions):
    state_PT = state_range_PT[point]
    output = policy(state_PT)
    output_np = output.data.numpy()
    pretrained_policy_control[point] = output_np[0]

# plot evolution of states and controls
time_array = [ti + div * dtime for div in range(divisions)]
plot_state_policy_evol(time_array, y1_PT, y2_PT, pretrained_policy_control,
                       objective=pretraining_objective)

# ---------------------------------------------------
#                  MAIN LOOP start
# ---------------------------------------------------

# problem parameters
episodes = 100000
records = 1000
std_sqr_red = 0.9

std_sqr = 1.0  # remember this is reduced first iter !!
episode_update_n = 1  # every how many episodes I update !!
rewards = [None for i in range(episode_update_n)]  # keep track of rewards per episode
log_probs = [[None for j in range(divisions)] for i in range(episode_update_n)]
epi_n = -1  # because I sum one
rewards_record = []

optimizer = optim.Adam(policy.parameters(), lr=1e-2)

# create folders to save pictures and model
os.makedirs('figures', exist_ok=True)
os.makedirs('serializations', exist_ok=True)

for episode in range(episodes):
    # COMPUTE EPISODE FOR LATER REINFORCEMENT LEARNING

    if episode % records == 0:  # diminish std to focus more on explotation
        std_sqr = std_sqr * std_sqr_red

    # run episode with network policy
    initial = np.array([1, 0])
    compute_run(
        policy, initial, params, log_probs, rewards, epi_n,
        dtime, divisions, ti, tf, std_sqr,
        return_evolution=False
        )
    epi_n = epi_n + 1

    # TRAIN CURRENT POLICY NETWORK USING COMPUTED TRAJECTORIES
    if episode != 0 and episode % episode_update_n == 0:
        gamma = 0.99
        loss = 0
        for i_ep in range(episode_update_n):
            R = torch.zeros(1)  # , 1)
            for i_j in reversed(range(divisions)):  # not sure why
                R = gamma * R + rewards[i_ep]
                loss = loss - (log_probs[i_ep][i_j] * R.expand_as(log_probs[i_ep][i_j])).sum()
                # A3C: We add entropy to the loss to encourage exploration
                # https://github.com/dennybritz/reinforcement-learning/issues/34

        loss = loss / episode_update_n
        optimizer.zero_grad()
        loss.backward()
        # utils.clip_grad_norm(policy.parameters(), 40)
        # https://discuss.pytorch.org/t/about-torch-nn-utils-clip-grad-norm/13873/2
        optimizer.step()
        # define new rewards and log probs for next episodes to train (like a "zero_grad")
        epi_n = 0
        rewards = [None for i in range(episode_update_n)]
        log_probs = [[None for j in range(divisions)] for i in range(episode_update_n)]
        
    # PLOT CURRENT MEAN POLICY
    if episode % records == 0:
        y1_l, y2_l, U_l = compute_run(
            policy, np.array([1, 0]), params, log_probs, rewards, epi_n,
            dtime, divisions, ti, tf, std_sqr,
            return_evolution=True
            )

        print(f'episode = {episode}\tcurrent_reward = {y2_l[-1]:0.3f}')
        rewards_record.append(y2_l[-1])
        print('std_sqr = ', std_sqr)

        store_path = join('figures', f'profile_episode_{str(episode)}_REINFORCE.png')
        plot_state_policy_evol(
            time_array, y1_l, y2_l, U_l, show=False, store_path=store_path
            )
        plt.close()

        torch.save(policy, join('serializations', 'policy.pt'))

print('finished learning')
print('std_sqr = ', std_sqr)
plt.plot(rewards_record)
plt.ylabel('reward value')
plt.show()
