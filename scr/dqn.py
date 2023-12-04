import copy
import math
from collections import namedtuple

import gym
import matplotlib.pyplot as plt
import more_itertools as mitt
import numpy as np
import random
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from tqdm.auto import tqdm, trange


class ExponentialSchedule:
    def __init__(self, value_from, value_to, num_steps):
        """Exponential schedule from `value_from` to `value_to` in `num_steps` steps.

        $value(t) = a \exp (b t)$

        :param value_from: initial value
        :param value_to: final value
        :param num_steps: number of steps for the exponential schedule
        """
        self.value_from = value_from
        self.value_to = value_to
        self.num_steps = num_steps

        # Calculate a
        self.a = value_from / math.exp(0)  # value(t=0) = a * exp(b * 0) = a * exp(0) = a * 1 = a
        self.b = math.log(value_to / self.a) / (num_steps - 1)  # b = ln(value(t=n) / a) / n
        # print("a =", self.a, "b =", self.b)

    def value(self, step) -> float:
        """Return exponentially interpolated value between `value_from` and `value_to`interpolated value between.

        returns {
            `value_from`, if step == 0 or less
            `value_to`, if step == num_steps - 1 or more
            the exponential interpolation between `value_from` and `value_to`, if 0 <= steps < num_steps
        }

        :param step:  The step at which to compute the interpolation.
        :rtype: float.  The interpolated value.
        """

        if step <= 0:
            value = self.value_from
        elif step >= self.num_steps - 1:
            value = self.value_to
        else:
            value = self.a * math.exp(self.b * step)

        return value


# Batch namedtuple, i.e. a class which contains the given attributes
Batch = namedtuple(
    'Batch', ('states', 'actions', 'rewards', 'next_states', 'dones')
)


class ReplayMemory:
    def __init__(self, max_size, state_size):
        """Replay memory implemented as a circular buffer.

        Experiences will be removed in a FIFO manner after reaching maximum
        buffer size.

        Args:
            - max_size: Maximum size of the buffer.
            - state_size: Size of the state-space features for the environment.
        """
        self.max_size = max_size
        self.state_size = state_size

        # preallocating all the required memory, for speed concerns
        self.states = torch.empty((max_size, state_size))
        self.actions = torch.empty((max_size, 1), dtype=torch.long)
        self.rewards = torch.empty((max_size, 1))
        self.next_states = torch.empty((max_size, state_size))
        self.dones = torch.empty((max_size, 1), dtype=torch.bool)

        # pointer to the current location in the circular buffer
        self.idx = 0
        # indicates number of transitions currently stored in the buffer
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.

        :param state:  1-D np.ndarray of state-features.
        :param action:  integer action.
        :param reward:  float reward.
        :param next_state:  1-D np.ndarray of state-features.
        :param done:  boolean value indicating the end of an episode.
        """

        # YOUR CODE HERE:  store the input values into the appropriate
        # attributes, using the current buffer position `self.idx`
        self.states[self.idx] = torch.tensor(state)
        self.actions[self.idx] = torch.tensor(action)
        self.rewards[self.idx] = torch.tensor(reward)
        self.next_states[self.idx] = torch.tensor(next_state)
        self.dones[self.idx] = torch.tensor(done)

        # DO NOT EDIT
        # circulate the pointer to the next position
        self.idx = (self.idx + 1) % self.max_size
        # update the current buffer size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> Batch:
        """Sample a batch of experiences.

        If the buffer contains less that `batch_size` transitions, sample all
        of them.

        :param batch_size:  Number of transitions to sample.
        :rtype: Batch
        """

        # YOUR CODE HERE:  randomly sample an appropriate number of
        # transitions *without replacement*.  If the buffer contains less than
        # `batch_size` transitions, return all of them.  The return type must
        # be a `Batch`.

        sample_indices = torch.randint(0, self.size, (batch_size,))
        batch = Batch(
            states=self.states[sample_indices],
            actions=self.actions[sample_indices],
            rewards=self.rewards[sample_indices],
            next_states=self.next_states[sample_indices],
            dones=self.dones[sample_indices]
        )

        return batch

    def populate(self, env, num_steps):
        """Populate this replay memory with `num_steps` from the random policy.

        :param env:  Openai Gym environment
        :param num_steps:  Number of steps to populate the
        """

        # YOUR CODE HERE:  run a random policy for `num_steps` time-steps and
        # populate the replay memory with the resulting transitions.
        # Hint:  don't repeat code!  Use the self.add() method!
        state = env.reset()
        for _ in range(num_steps):
            action = random.choice(env.getLegalActions())   # env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            action_int = env.get_int(action)

            self.add(state, action_int, reward, next_state, done)
            if done:
                state = env.reset()
            else:
                state = next_state

#
# class DQN(nn.Module):
#     def __init__(self, state_dim, action_dim, *, num_layers=3, hidden_dim=256):
#         """Deep Q-Network PyTorch model.
#
#         Args:
#             - state_dim: Dimensionality of states
#             - action_dim: Dimensionality of actions
#             - num_layers: Number of total linear layers
#             - hidden_dim: Number of neurons in the hidden layers
#         """
#
#         super().__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#
#         # * there are `num_layers` nn.Linear modules / layers
#         self.layers = nn.ModuleList()
#
#         # * all activations except the last should be ReLU activations
#         #   (this can be achieved either using a nn.ReLU() object or the nn.functional.relu() method)
#         for _ in range(num_layers - 1):
#             self.layers.append(nn.Linear(state_dim, hidden_dim))
#             self.layers.append(nn.ReLU())
#             state_dim = hidden_dim  # Update state_dim for subsequent layers
#
#         # * the last activation can either be missing, or you can use nn.Identity()
#         self.layers.append(nn.Linear(hidden_dim, action_dim))
#
#     def forward(self, states) -> torch.Tensor:
#         """Q function mapping from states to action-values.
#
#         :param states: (*, S) torch.Tensor where * is any number of additional
#                 dimensions, and S is the dimensionality of state-space.
#         :rtype: (*, A) torch.Tensor where * is the same number of additional
#                 dimensions as the `states`, and A is the dimensionality of the
#                 action-space.  This represents the Q values Q(s, .).
#         """
#         # Pass states through layers
#         for layer in self.layers:
#             states = layer(states)
#
#         return states
#
#     # utility methods for cloning and storing models.  DO NOT EDIT
#     @classmethod
#     def custom_load(cls, data):
#         model = cls(*data['args'], **data['kwargs'])
#         model.load_state_dict(data['state_dict'])
#         return model
#
#     def custom_dump(self):
#         return {
#             'args': (self.state_dim, self.action_dim),
#             'kwargs': {
#                 'num_layers': self.num_layers,
#                 'hidden_dim': self.hidden_dim,
#             },
#             'state_dict': self.state_dict(),
#         }
#
#
# def train_dqn_batch(optimizer, batch, dqn_model, dqn_target, gamma) -> float:
#     """Perform a single batch-update step on the given DQN model.
#
#     :param optimizer: nn.optim.Optimizer instance.
#     :param batch:  Batch of experiences (class defined earlier).
#     :param dqn_model:  The DQN model to be trained.
#     :param dqn_target:  The target DQN model, ~NOT~ to be trained.
#     :param gamma:  The discount factor.
#     :rtype: float  The scalar loss associated with this batch.
#     """
#
#     # given models and the batch of data.
#     states = batch.states
#     actions = batch.actions
#     rewards = batch.rewards
#     next_states = batch.next_states
#     dones = batch.dones
#
#     # compute the values and target_values tensors using batch inputs.
#     values = dqn_model(states)
#
#     # Compute Q-values for the next states using dqn_target
#     next_values = dqn_target(next_states)
#
#     # Select Q-values for the actions taken
#     # Assuming actions is the index tensor and values is the tensor of Q-values
#     predicted_qvalues_for_actions = values[range(len(actions)), actions.squeeze()]
#     values = predicted_qvalues_for_actions.unsqueeze(1)
#
#     # compute V*(next_states) using predicted next q-values
#     next_state_values = torch.max(next_values, dim=1)[0]
#     #     print("next_state_values", next_state_values)
#
#     next_state_values = next_state_values * (1.0 - dones.squeeze().float())
#
#     target_qvalues_for_actions = rewards + gamma * next_state_values.unsqueeze(1)
#     #     print("target_qvalues_for_actions", target_qvalues_for_actions)
#     target_values = target_qvalues_for_actions.detach()
#     #     print("target_values not gradient", target_values)
#     #     print("values",values)
#
#     # DO NOT EDIT FURTHER
#
#     assert (
#             values.shape == target_values.shape
#     ), 'Shapes of values tensor and target_values tensor do not match.'
#
#     # testing that the value tensor requires a gradient,
#     # and the target_values tensor does not
#     assert values.requires_grad, 'values tensor should not require gradients'
#     assert (
#         not target_values.requires_grad
#     ), 'target_values tensor should require gradients'
#
#     # computing the scalar MSE loss between computed values and the TD-target
#     loss = F.mse_loss(values, target_values)
#
#     optimizer.zero_grad()  # reset all previous gradients
#     loss.backward()  # compute new gradients
#     optimizer.step()  # perform one gradient descent step
#
#     return loss.item()
#
# def train_dqn(
#         env,
#         num_steps,
#         *,
#         num_saves=5,
#         replay_size,
#         replay_prepopulate_steps=0,
#         batch_size,
#         exploration,
#         gamma,
# ):
#     """
#     DQN algorithm.
#
#     Compared to previous training procedures, we will train for a given number
#     of time-steps rather than a given number of episodes.  The number of
#     time-steps will be in the range of millions, which still results in many
#     episodes being executed.
#
#     Args:
#         - env: The openai Gym environment
#         - num_steps: Total number of steps to be used for training
#         - num_saves: How many models to save to analyze the training progress.
#         - replay_size: Maximum size of the ReplayMemory
#         - replay_prepopulate_steps: Number of steps with which to prepopulate the memory
#         - batch_size: Number of experiences in a batch
#         - exploration: a ExponentialSchedule
#         - gamma: The discount factor
#
#     Returns: (saved_models, returns)
#         - saved_models: Dictionary whose values are trained DQN models
#         - returns: Numpy array containing the return of each training episode
#         - lengths: Numpy array containing the length of each training episode
#         - losses: Numpy array containing the loss of each training batch
#     """
#     # check that environment states are compatible with our DQN representation
#     assert (
#             isinstance(env.observation_space, gym.spaces.Box)
#             and len(env.observation_space.shape) == 1
#     )
#
#     # get the state_size from the environment
#     state_size = env.observation_space.shape[0]
#
#     print("state_size:", state_size, "env.action_space.n", env.action_space.n)
#
#     # initialize the DQN and DQN-target models
#     dqn_model = DQN(state_size, env.action_space.n)
#     dqn_target = DQN.custom_load(dqn_model.custom_dump())
#
#     # initialize the optimizer
#     optimizer = torch.optim.Adam(dqn_model.parameters())
#
#     # initialize the replay memory and prepopulate it
#     memory = ReplayMemory(replay_size, state_size)
#     memory.populate(env, replay_prepopulate_steps)
#
#     # initiate lists to store returns, lengths and losses
#     rewards = []
#     returns = []
#     lengths = []
#     losses = []
#
#     # initiate structures to store the models at different stages of training
#     t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)
#     saved_models = {}
#
#     i_episode = 0  # use this to indicate the index of the current episode
#     t_episode = 0  # use this to indicate the time-step inside current episode
#
#     state = env.reset()  # initialize state of first episode
#     # iterate for a total of `num_steps` steps
#     pbar = trange(num_steps)
#     for t_total in pbar:
#         # use t_total to indicate the time-step from the beginning of training
#
#         # save model
#         if t_total in t_saves:
#             model_name = f'{100 * t_total / num_steps:04.1f}'.replace('.', '_')
#             saved_models[model_name] = copy.deepcopy(dqn_model)
#
#         # YOUR CODE HERE:
#         #  * sample an action from the DQN using epsilon-greedy
#         # action = epsilon_greedy_action(dqn_model, state, exploration.value(t_total))
#
#         if random.random() < exploration.value(t_total):
#             # Explore: Choose a random action from the action space
#             action = random.randint(0, dqn_model.action_dim - 1)
#         else:
#             # Exploit: Choose the action with the highest Q-value from the model
#             with torch.no_grad():
#                 state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
#                 q_values = dqn_model(state_tensor)  # Get Q-values for the state
#                 action = q_values.argmax().item()  # Choose action with the highest Q-value
#
#         #  * use the action to advance the environment by one step
#         next_state, reward, done, _ = env.step(action)
#         #  print("debug:", t_total, state, action, next_state, reward, done)
#
#         #  * store the transition into the replay memory
#         memory.add(state, action, reward, next_state, done)
#
#         # Once every 4 steps,
#         if (t_total + 1) % 4 == 0:
#             #  * sample a batch from the replay memory
#             batch = memory.sample(batch_size)
#
#             #  * perform a batch update (use the train_dqn_batch() method!)
#             loss = train_dqn_batch(optimizer, batch, dqn_model, dqn_target, gamma)
#             losses.append(loss)
#
#         # Once every 10_000 steps,
#         if (t_total + 1) % 10000 == 0:
#             # update the target network (use the dqn_model.state_dict() and dqn_target.load_state_dict() methods!)
#             dqn_target.load_state_dict(dqn_model.state_dict())
#
#         if done:
#             # YOUR CODE HERE:  anything you need to do at the end of an
#             # episode, e.g. compute return G, store stuff, reset variables,
#             # indices, lists, etc.
#
#             # Calculate G (cumulative reward)
#             G = sum(rewards)  # Assuming episode_rewards contains rewards received in the episode
#             returns.append(G)
#
#             # Calculate eps (exploration rate)
#             eps = exploration.value(t_episode)  # Get epsilon value at the end of the episode
#             lengths.append(t_episode)
#
#             pbar.set_description(
#                 f'Episode: {i_episode} | Steps: {t_episode + 1} | Return: {G:5.2f} | Epsilon: {eps:4.2f}'
#             )
#
#             # Reset environment for the next episode and update necessary variables
#             state = env.reset()
#             t_episode = 0
#             i_episode += 1
#             episode_rewards = 0
#             rewards = []
#
#         else:
#             # YOUR CODE HERE:  anything you need to do within an episode
#             state = next_state
#             t_episode += 1
#             rewards.append(reward)
#
#     saved_models['100_0'] = copy.deepcopy(dqn_model)
#
#     return (
#         saved_models,
#         np.array(returns),
#         np.array(lengths),
#         np.array(losses),
#     )
#
#
# def run_dqn(env):
#     gamma = 0.99
#
#     # we train for many time-steps;  as usual, you can decrease this during development / debugging.
#     # but make sure to restore it to 1_500_000 before submitting.
#     num_steps = 1_500_000
#     num_saves = 5  # save models at 0%, 25%, 50%, 75% and 100% of training
#
#     replay_size = 200_000
#     replay_prepopulate_steps = 50_000
#
#     batch_size = 64
#     exploration = ExponentialSchedule(1.0, 0.01, 1_000_000)
#
#     # this should take about 90-120 minutes on a generic 4-core laptop
#     dqn_models, returns, lengths, losses = train_dqn(
#         env,
#         num_steps,
#         num_saves=num_saves,
#         replay_size=replay_size,
#         replay_prepopulate_steps=replay_prepopulate_steps,
#         batch_size=batch_size,
#         exploration=exploration,
#         gamma=gamma,
#     )
#
#     assert len(dqn_models) == num_saves
#     assert all(isinstance(value, DQN) for value in dqn_models.values())
#
#     # saving computed models to disk, so that we can load and visualize them later.
#     checkpoint = {key: dqn.custom_dump() for key, dqn in dqn_models.items()}
#     torch.save(checkpoint, f'checkpoint_{env.spec.id}.pt')
#
#     return dqn_models