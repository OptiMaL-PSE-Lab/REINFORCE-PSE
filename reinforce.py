import os
from os.path import join
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import model_integrator
from policies import PolicyNetwork
from utilities import normal_torch, pretraining, select_action, compute_run

np.random.seed(seed=666)
torch.manual_seed(666)
pi = Variable(torch.FloatTensor([np.pi]))


# problem parameters
episode_n = 100000
record_n = 1000
std_sqr_red = (1, record_n)

# specifications for dynamic system
dtime = 0.1  # define time step
iter_n = 0  # keep track of number of integration iterations
ti = 0  # define initial time
tf = 1  # define final time
t_steps = int((tf - ti) / dtime)  # number of steps per episode
params = {'a_p': 0.5, 'b_p': 1}

# define policy network and other learning/reporting parameters
policy = PolicyNetwork(6, 3, 1)  # define ANN with 6 hidden, 3 input, 1 output
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
All_rewards_l = []
std_sqr = 1.0  # remember this is reduced first iter !!
episode_update_n = 1  # every how many episodes I update !!
rewards_l = [None for i in range(episode_update_n)]  # keep track of rewards per episode
log_probs_l = [[None for j in range(t_steps)] for i in range(episode_update_n)]
epi_n = -1  # because I sum one


# Pre-training
inputs_PT = [i_PT * 5.0 / t_steps for i_PT in range(t_steps)]
runs_PT = 100
pert_size = 0.1
y1_PT, y2_PT, t_PT, U_u_PT = pretraining(
    policy, inputs_PT, params, runs_PT, pert_size, 
    t_steps, ti, tf, dtime, 
    initial_state_I=np.array([1, 0])
    )
y1_PT_Torch = Variable(torch.Tensor(y1_PT))
y2_PT_Torch = Variable(torch.Tensor(y2_PT))
t_PT_Torch = Variable(torch.Tensor(t_PT))

U_PT_plot = [None for i_PT in range(t_steps)]
for point in range(t_steps):
    # evaluate model with real data
    inp_point = Variable(torch.Tensor(
        [y1_PT_Torch[point], y2_PT_Torch[point], t_PT_Torch[point]]))
    output = policy(inp_point)
    output_np = output.data.numpy()
    # compile results
    U_PT_plot[point] = output_np[0]

plt.subplot2grid((2, 4), (0, 0), colspan=2)
plt.plot(y1_PT)
plt.ylabel('y1 ', rotation=360, fontsize=15)
plt.xlabel('time', fontsize=15)

plt.subplot2grid((2, 4), (0, 2), colspan=2)
plt.plot(y2_PT)
plt.ylabel('y2 ', rotation=360, fontsize=15)
plt.xlabel('time', fontsize=15)

plt.subplot2grid((2, 4), (1, 0), colspan=2)
plt.plot(U_PT_plot)
plt.ylabel('u', rotation=360, fontsize=15)
plt.xlabel('time', fontsize=15)

plt.subplot2grid((2, 4), (1, 2), colspan=2)
plt.plot(U_PT_plot)
plt.plot(U_u_PT, 'ro')
plt.ylabel('u', rotation=360, fontsize=15)
plt.xlabel('time', fontsize=15)
plt.show()

# ---------------------------------------------------
#                  MAIN LOOP start
# ---------------------------------------------------

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
        dtime, t_steps, ti, tf, std_sqr, epi_n,
        plot_CR=False
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
                    log_probs_l[i_ep][i_j] * Variable(R).expand_as(log_probs_l[i_ep][i_j])
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
        log_probs_l = [[None for j in range(t_steps)] for i in range(episode_update_n)]
    # PLOT CURRENT MEAN POLICY
    if i_episode % record_n == 0:
        r_report, y1_l, y2_l, t_l, U_u_l = compute_run(
            policy, np.array([1, 0]), params, log_probs_l,
            dtime, t_steps, ti, tf, std_sqr, epi_n,
            plot_CR=True
            )

        print('i_episode = ', i_episode, '     current_reward = ', r_report)
        All_rewards_l.append(r_report)
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
        plt.plot(U_u_l)
        plt.ylabel('u', rotation=360, fontsize=15)
        plt.xlabel('time', fontsize=15)
        plt.savefig(
            join('figures', 'profile_iter_a_' + str(i_episode) + '_REINFORCE_v3.png')
            )
        plt.close()
        torch.save(policy, join('serializations', 'ann_pretrain_10-6.pt'))

print('finished learning')
print('std_sqr = ', std_sqr)
plt.plot(All_rewards_l)
plt.ylabel('reward value')
plt.show()