# RL for batch processes optimization

The basic idea is the usage of reinforcement learning techniques to optimize the controls
taken over a chemical system evolution to optimize the outcome of certain variables.
Current models of batch processes are ODE systems with certain free parameters as controls.

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
  * PPO with baseline over mean reward as advantage function.

### Action evolution sample

![Original](https://i.imgur.com/z9CPjA3.gif)

<html>
    <body>
        <p><img src="https://i.imgur.com/z9CPjA3.gif" alt="Action Probability Distribution" width="800" height="400" align="middle"></p>
    </body>
</html>
