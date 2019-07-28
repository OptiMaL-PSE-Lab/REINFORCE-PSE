# Reinforcement Learning for batch processes optimization

We use reinforcement learning techniques to optimize continuum controls over chemical systems
modelled via ODE systems. The controls are free parameters of the ODE system and the reward
is a variable of the system, which represents a byproduct of the chemical reaction over time.

## Surrogate ODE systems

ODE systems represent a surrogate approximation of real-word chemical systems.
Closer approximations to reality involve more complex surrogate models.
Here we study an approach to optimize the controls of a system over a simple ODE model to later ease the study of optimal controls over a more complex yet closely related ODE model.

The objective is to find the optimal controls $`U_1(t)`$ and $`U_2(t)`$ contrained to the interval $`[0,5]`$ that maximize the final value of the byproduct $`y_2`$ over the time span $`t \in (0, 1)`$.

The initial conditions are fixed: $`y_1 = 1`$, $`y_2 = 0`$

### Simple system

$` \dot{y_1} = -(U_1 + a  U_1 ^ 2)  y_1 + d  U_2 `$

$` \dot{y_2} = (b  U_1 - c  U_2)  y_1 `$

**Parameters**: $` a, b, c, d = 0.5, 1.0, 0.7, 0.5 `$

### Complex system

$` \dot{y_1} = -(U_1 + a  U_2 ^ 2)  y_1 + d  U_2  y_2 / (y_1 + y_2) `$

$` \dot{y_2} = (b  U_1 - c  U_2)  y_1 `$

**Parameters**: $` a, b, c, d = 0.5, 1.0, 1.0, 1.0 `$

## Policy gradients

We use the classic REINFORCE policy gradient algorithm and the recent Proximal Policy Optimization (PPO) variation to optimize the controls via stochastic sampling of trayectories.
We use stochastic continuum distributions to deal with continuum controls.
In particular, we use the Beta distribution to enforce interval constraints over the controls.

## Implementation details

* The evolution of the model between subsequent states corresponds to the integration of the ODE system over a fixed fraction of time.
* Each state is composed of the current model variables.
* Policies take states and return the mean and variance of a probability distribution over the possible continuous actions available at each time.
  * An affine Beta distribution over the acceptable interval $`U \in [0,5]`$.
* Not too deep Neural Networks serve as policies.
  * Simple Neural Networks include in the states the time left.
  * Recurrent Neural Netwoks take previous hidden states instead.
* Several sampled episodes are needed to estimate the loss function to be optimized.
  * REINFORCE algorithm with mean reward baseline.
  * PPO with deviation from mean reward as advantage function.

Transfer learning techniques help to leverage the inner weights if the policy learned in the simpler model to ease the training needed for the related but more complex model.

## Code execution

Run `python main.py` to execute the whole logic:

* Pretrain policy with linearly increasing controls over time.
* Train policy for large iterations over simpler model.
* Freeze inner weights of policies and retrain last layers with complex model with fewer iterations.

Current reward is printed over console and relevant profiles are stored in a `./figures/` subdirectory.

Main parameters (more available in command line interface from `main.py`):

* method: 'ppo' or 'reinforce'
* episode-batch: number of sample episodes run to estimate loss function
* chained-steps: gradient descent steps taken after episode sampling (usually 1 for REINFORCE and ~5 for PPO)
* iterations: repetitions of sampling and optimize step

### Evolution of action distributions of sampled episodes

<img src="https://i.imgur.com/HMf0out.mp4" alt="Simple model" align="middle">

<img src="https://i.imgur.com/J7NE5o1.mp4" alt="Complex model (via transfer learning)" align="middle">

### Reward evolution

<img src="https://i.imgur.com/AME2Gyz.mp4" alt="Simple model" align="middle">

<img src="https://i.imgur.com/t9RiibJ.mp4" alt="Complex model (via transfer learning)" align="middle">

**Kudos**: gifs made with the blazing-fast [gifski](https://github.com/ImageOptim/gifski) and transformed via [ImageOptim API](https://imageoptim.com/api/ungif)!
