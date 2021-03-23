# Reinforcement Learning for batch processes optimization

We use reinforcement learning techniques to optimize continuum controls over chemical systems
modelled via ODE systems. The controls are free parameters of the ODE system and the reward
is a variable of the system after some time of evolution of the system, which represents a byproduct of the chemical reaction.

## ODE systems models

ODE systems represent an approximation of real-word chemical systems.
Here we study an approach to optimize the controls of a system over a simple ODE model to later ease the study of optimal controls over a more complex yet closely related ODE model.

The objective is to find the optimal controls $`U_1(t)`$ and $`U_2(t)`$ contrained to the interval $`[0,5]`$ that maximize the final value of the byproduct $`y_2`$ at the end of the time span $`t \in (0, 1)`$.

The initial state is fixed: $`y_1 = 1`$, $`y_2 = 0`$

### Simple system

$` \dot{y_1} = -(U_1 + \alpha  U_1 ^ 2)  y_1 + \omega  U_2 `$

$` \dot{y_2} = (\beta  U_1 - \gamma  U_2)  y_1 `$

**Parameters**: $` \alpha, \beta, \gamma, \omega = 0.5, 1.0, 0.7, 0.5 `$

### Complex system

$` \dot{y_1} = -(U_1 + \alpha  U_2 ^ 2)  y_1 + \omega  U_2  y_2 / (y_1 + y_2) `$

$` \dot{y_2} = (\beta  U_1 - \gamma  U_2)  y_1 `$

**Parameters**: $` \alpha, \beta, \gamma, \omega = 0.5, 1.0, 1.0, 1.0 `$

## Policy gradients

We use the classic REINFORCE policy gradient algorithm with a baseline, and the recent Proximal Policy Optimization (PPO) variation, to optimize the controls via stochastic sampling of trayectories.
We use the Beta distribution to sample continuum controls to enforce interval constraints over the controls.

## Implementation details

* The evolution of the model between subsequent states corresponds to the integration of the ODE system over a fixed fraction of time.
* Each state is composed of the current ODE variables.
* Policies take states and return the mean and variance of a probability distribution over the possible continuous actions available at each time.
  * An affine Beta distribution over the acceptable interval $`U \in [0,5]`$.
* Not too deep Neural Networks serve as policies.
  * Simple Neural Networks include in the states the time left until $t=1$.
  * Recurrent Neural Netwoks take previous hidden states instead.
* Several sampled episodes are used to estimate the policy gradient.
  * REINFORCE algorithm with mean reward baseline.
  * PPO with deviation from mean reward as advantage function.
* Starting policy is pretrained to follow a random Chebyshev polynomial on the mean with fixed variance.

A simple transfer learning technique is explored to leverage the inner weights of the policy learned in the simpler model to ease the training needed for the related but more complex model.

## Code execution

A conda environment file is provided for easy reproduction of results:
```
conda env create -f environment.yml
conda activate batch-reactor
```

Run `python src/main.py` to execute the whole logic (use `--help` for a list of available parameters):

* Pretrain policy to yield predefined function forms that satisfy the constraints.
* Train policy for large iterations over simpler model.
* Freeze inner weights of policies and retrain last layers with complex model with fewer iterations.

Relevant data and plots are stored in `results/_execution_datetime_/...`. Hyperparameters of the run are stored in a yaml config file inside.

Main parameters (more available in command line interface from `main.py`) are:

* method: 'ppo' or 'reinforce'
* episode-batch: number of sample episodes run to estimate loss function
* chained-steps: gradient descent steps taken after episode sampling (usually 1 for REINFORCE and ~3 for PPO)
* iterations: repetitions of sampling and optimize step

### Evolution of action distributions of sampled episodes

![Simple Model](https://user-images.githubusercontent.com/12092488/112073403-a5716180-8b39-11eb-8b1c-55932adb686f.gif)

![Complex Model](https://user-images.githubusercontent.com/12092488/112073235-4ad80580-8b39-11eb-9698-bf1bff7b8090.gif)

### Reward evolution

![Simple Model](https://user-images.githubusercontent.com/12092488/112073315-778c1d00-8b39-11eb-9ff5-0cf9717f782e.gif)

![Complex Model](https://user-images.githubusercontent.com/12092488/112073364-912d6480-8b39-11eb-850b-ebc642679180.gif)

**Kudos**: gifs made with the blazing-fast [gifski](https://github.com/ImageOptim/gifski) and transformed via [ImageOptim API](https://imageoptim.com/api/ungif)!
