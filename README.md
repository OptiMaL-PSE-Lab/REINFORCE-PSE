# Reinforcement Learning for batch processes optimization

The basic idea is the usage of reinforcement learning techniques to optimize the controls
taken over a chemical system evolution to optimize the outcome of certain variables.
Current models of batch processes are ODE systems with certain free parameters as controls.

## Systems modelled

* For the following ODE system with parameters $`a = 1/2`$, $`b = 1`$ and control $`U(t) \in [0,5]`$,
    maximize the value of $`y_2(t = 1)`$ from the initial state $`y_1 = 1, y_2 = 0`$ at $`t=0`$.

```math
    \dot{y_1} = -(U + U^2 a) y_1
    \dot{y_2} = U y_1 b
```

## Implementation

* The evolution of the model corresponds to the integration of the ODE system over a fixed
  amount of time.
* Each state is composed of the current model variables and the remaining time of the process.
* Policies take states and return the mean and variance of a probability distribution over
  the possible continuous actions available at each time.
  * Beta probability distributions are used to deal with constrained continuous actions.
* Gradient policy methods are used to optimize stochastic policies results.
  Several episode samples are needed to build the loss function to optimize over.
  * REINFORCE algorithm with mean reward baseline.
  * PPO with deviation from mean reward as advantage function.

## Usage

Run `python main.py`, it prints to console the current reward and
stores the relevant profiles in a `./figures/` subdirectory (not tracked via git).

Parameters:

* method: 'ppo' or 'reinforce'
* episode_batch: number of sample episodes run to estimate loss function
* epochs: gradient descent steps taken after episode sampling (just 1 for REINFORCE)
* iterations: sampling-optimizing repetitions

### Reward evolution

* REINFORCE

<img src="https://i.imgur.com/3BQDNKp.png"  width="600" height="400">

* PPO

<img src="https://i.imgur.com/65JxgQ1.png"  width="600" height="400">

### Action distributions of sampled episodes

* REINFORCE

<img src="https://i.imgur.com/DMOoLJc.gif" alt="Action Probability Distribution" width="1000" height="500" align="middle">

* PPO

<img src="https://i.imgur.com/ECyvYGz.gif" alt="Action Probability Distribution" width="1000" height="500" align="middle">

**Kudos**: gifs made with the blazing-fast [gifski](https://github.com/ImageOptim/gifski)!
