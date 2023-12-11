# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from featureExtractors import *
from tqdm import trange
from dqn import ExponentialSchedule
import random
import util
from collections import defaultdict
from typing import Sequence, Callable, Tuple
import math
import numpy as np

class QLearningAgent:
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        #ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        #print(args)
        self.Q = util.Counter()
        self.epsilon = args['epsilon']
        self.dimension = args['dimension']
        self.discount = args['gamma']
        self.alpha = args['alpha']
        print(self.epsilon, self.dimension, self.discount, self.alpha)

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #print(state, action)
        if self.Q[(state, action)] == None:
            self.Q[(state, action)] = 0.0
            
        return self.Q[(state, action)]
        

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"  
        legalActions = self.getLegalActions(state)

        if legalActions == [] or state == 'TERMINAL_STATE':
            return 0.0

        qval = [self.getQValue(state, action) for action in legalActions]

        if qval == []:
            print("Error! ", qval, legalActions)

        return max(qval)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        ans = []
        legalActions = self.getLegalActions(state)
        
        if legalActions == [] or state == 'TERMINAL_STATE':
            return None

        max = self.computeValueFromQValues(state)  # To pick max from Q value states

        for action in legalActions:
            if max == self.getQValue(state, action):
                ans.append(action)

        if ans == []:
            # print("No Q-value! Choose random. ", max, legalActions)
            ans.append(random.choice(legalActions))

        return random.choice(ans)  # To pick randomly from a list in the maximum options


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        legalActions = self.getLegalActions(state)
        if legalActions == []:
            print("TERMINAL_STATE!!!!", state)
            return None

        if state.getPlayerTurn() == 0:
            return state.getCenter()
        elif util.flipCoin(self.epsilon): #To pick randomly from a list, especially in beginning
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        #print("=======================update===========================")
        #print(state, action, nextState, reward)
        self.Q[(state, action)] = self.Q[(state, action)] + self.alpha * (reward + self.discount * self.getValue(nextState) - self.Q[(state, action)])
        #print(self.Q[(state, action)])

    def explorationProb(self, eps):
        self.epsilon = eps
        # print("Update! explorationProb:", self.epsilon)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getLegalActions(self, state):
        matrix = state.matrix
        #print(matrix)
        actions = [(x, y) for x in range(self.dimension) for y in range(self.dimension) if matrix[x][y] == 0]
        #print("actions:", actions)

        return actions



class OmokQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.5, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        #self.doAction(state,action)
        return action


class ApproximateQAgent(OmokQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        OmokQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        sum = 0
        featureVector = self.featExtractor.getFeatures(state, action)
        # learns weights for features of states, where many states might share the same features.
        for f in featureVector:
            sum += self.weights[f] * featureVector[f]

        if sum > 100000:
            print(sum) # Debugging!

        return sum

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        featureVector = self.featExtractor.getFeatures(nextState, action)

        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        # update your weight vectors similarly to how you updated Q-values:
        # print(difference)
        # print(featureVector)

        for f in featureVector:
            self.weights[f] = self.weights[f] + self.alpha * difference * featureVector[f]
            # print(f, self.weights[f])

        # print("test")

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        OmokQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

    def train(self, num_steps, gs):
        for i in trange(num_steps, desc="ApproximateQAgent Learning"):
            studying = 1
            #explore_rate = 0.9 - (i * 0.001)
            exploration = ExponentialSchedule(1.0, 0.01, num_steps) #1_000_000)
            self.explorationProb(exploration.value(i))

            while studying:
                prev_gs = gs.deepCopy()
                action = self.getAction(gs)
                row, col = action
                # print(row, col, gs.getPlayerTurn())
                gs.updatePlayerTurn(row, col)

                if gs.is_gameOver(row, col, gs.getColor(row, col)) == True:
                    # print(gs)

                    if gs.player_turn >= gs.dimension * gs.dimension:  # Draw
                        self.update(prev_gs, action, gs, 0)
                    elif gs.getColor(row, col) == 'white':  # Black lose
                        self.update(prev_gs, action, gs, -10)
                    else:
                        self.update(prev_gs, action, gs, 10)  # Black win
                    gs.reset()
                    self.featExtractor.resetFeatures()
                    studying += 1
                else:
                    self.update(prev_gs, action, gs, -0.1)

                if studying > 100:  # TBD: How long are you studying?
                    studying = 0

            # print(gs.win_history, "Please wait. Now learning...", i + 1, "/ 100")
            # gs.win_history = [0, 0, 0]

    def train_with_MinMax(self, num_games, gs, qa, ma):
        for i in trange(num_games, desc="ApproximateQAgent Learning"):
            studying = 1
            # explore_rate = 0.9 - (i * 0.001)
            exploration = ExponentialSchedule(1.0, 0.01, num_games)
            self.explorationProb(exploration.value(i))

            while studying:
                prev_gs = gs.deepCopy()

                if gs.getPlayerTurn() % 2 == 0:  # RL turn
                    agent = qa
                else:
                    agent = ma

                action = agent.getAction(gs)
                col, row = action
                gs.updatePlayerTurn(col, row)

                if gs.is_gameOver(col, row, gs.getColor(col, row)) == True:
                    # print(gs)
                    # print(gs.win_history)
                    studying = 0

                    if gs.getPlayerTurn() >= gs.dimension * gs.dimension:  # Draw
                        qa.update(prev_gs, action, gs, 0)
                    elif gs.getPlayerTurn() % 2 == 1:  # RL turn
                        qa.update(prev_gs, action, gs, 100)
                    else:
                        qa.update(prev_gs, action, gs, -100)  # LOSE

                    gs.print_history()
                    gs.reset()
                else:
                    if gs.getPlayerTurn() % 2 == 1:  # RL turn
                        qa.update(prev_gs, action, gs, -1)


    def train_vs_AI(self, num_games, gs, qa, ma):
        for i in trange(num_games, desc="MCTS Learning"):
            done = 0
            # explore_rate = 0.9 - (i * 0.001)
            exploration = ExponentialSchedule(1.0, 0.01, num_games)
            self.explorationProb(exploration.value(i))

            while not done:
                prev_gs = gs.deepCopy()

                if gs.getPlayerTurn() % 2 == 0:
                    agent = ma
                else:  # RL turn
                    agent = qa

                action = agent.getAction(gs)
                col, row = action
                gs.updatePlayerTurn(col, row)
                done, reward = gs.getReward(col, row)
                if agent == qa:
                    qa.update(prev_gs, action, gs, reward)
                if done:
                    if reward == 0.5:
                        gs.winCount('draw')
                    elif agent == ma:
                        gs.winCount('black')
                    elif agent == qa:
                        gs.winCount('white')
                    gs.print_history()


    def print(self):
        """
           Debugging
        """
        print("last explorationProb:", self.epsilon)
        for key, value in self.weights.items():
            print(key, value)


class MCTSagent(OmokQAgent):
    """
       Monte Carlo Tree Search LearningAgent

       You should only have to overwrite getQValue and update.
       All other QLearningAgent functions should work as is.
    """
    def __init__(self, depth, exploration_weight, **args):
        OmokQAgent.__init__(self, **args)
        self.T = defaultdict()
        self.depth = depth
        self.exploration_weight = exploration_weight
        # self.Q = defaultdict()  # total reward of each node
        self.N = defaultdict()  # total visit count for each node

    def epsilon_policy(self, Q: defaultdict, epsilon: float) -> Callable:
        """Creates an epsilon soft policy from Q values.

        A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

        Args:
            Q (defaultdict): current Q-values
            epsilon (float): softness parameter
        Returns:
            get_action (Callable): Takes a state as input and outputs an action
        """
        # Get number of actions
        num_actions = len(Q[0])
        choices = []
        for i, _ in enumerate(range(num_actions)):
            choices.append(i)

        def get_action(state: Tuple) -> int:
            if np.random.random() < epsilon:
                action = np.random.choice(choices)  # Explore
            else:
                action = self.argmax(Q[state])  # Exploit

            return action

        return get_action

    def argmax(self, arr: Sequence[float]) -> int:
        """Argmax that breaks ties randomly

        Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

        Note: np.argmax returns the first index that matches the maximum,
            so we define this method to use in Epsilon Greedy and UCB agents.
        Args:
            arr: sequence of values
        """

        max_value = max(arr)
        max_index = [i for i, val in enumerate(arr) if max_value == val]
        return np.random.choice(max_index)

    def getAction(self, state, depth=10):
        times = 10
        for i in range(times):
            self.simulate(state, depth, 0)
        return self.computeActionFromQValues(state)

    def simulate(self, state, dept, done):
        "Roll out(simulate) using rollout policy pi0."

        if dept == 0 or done:
            return 0
        if state not in self.T:
            legalActions = self.getLegalActions(state)
            for action in legalActions:
                self.Q[(state, action)] = 0
                self.N[(state, action)] = 0
                # Q = defaultdict(lambda: np.zeros(env.action_space.n))
                # N = defaultdict(lambda: np.zeros(env.action_space.n))
            self.T[state] = state.matrix
            return self.rollout(state, dept, 0)

        action = self.selection(state)
        next_state, reward, done = state.generateSuccessor(action)
        q = reward + self.discount * self.simulate(next_state, dept-1, done)
        self.N[(state, action)] = self.N[(state, action)] + 1
        self.Q[(state, action)] = self.Q[(state, action)] + (q - self.Q[(state, action)]) / self.N[(state, action)]
        return q

    def rollout(self, state, dept, done):
        if dept == 0 or done:
            return 0

        rollout_action = self.rolloutPolicy(state)
        next_state, reward, done = state.generateSuccessor(rollout_action)
        return reward + self.discount * self.rollout(next_state, dept-1, done)

    def selection(self, state):
        legalActions = self.getLegalActions(state)
        log_N_vertex = 0
        for action in legalActions:
            log_N_vertex += self.N[(state, action)]

        # Define a function to calculate the equation for each action
        def ucb_tree(action):
            if self.N[(state, action)] == 0:
                return float("inf")
            return self.Q[(state, action)] + self.exploration_weight * math.sqrt(log_N_vertex / self.N[(state, action)])

        # Get the action with maximum value using max function with a custom key
        return max(legalActions, key=ucb_tree)

    def rolloutPolicy(self, state):
        "Choose the best successor of node. (Choose a move in the game)"
        legalActions = self.getLegalActions(state)
        if legalActions == []:
            print("TERMINAL_STATE!!!!", state)
            return None

        if state.getPlayerTurn() == 0:
            return state.getCenter()
        else:  # the rollout is usually implemented following a uniform random policy
            return random.choice(legalActions)

    def train_vs_AI(self, num_games, gs, qa, ma):
        for i in trange(num_games, desc="MCTS Learning"):
            done = 0
            # explore_rate = 0.9 - (i * 0.001)
            exploration = ExponentialSchedule(1.0, 0.01, num_games)
            self.explorationProb(exploration.value(i))

            while not done:

                if gs.getPlayerTurn() % 2 == 0:
                    agent = ma
                else:  # RL turn
                    agent = qa

                action = agent.getAction(gs)
                col, row = action
                gs.updatePlayerTurn(col, row)
                done, reward = gs.getReward(col, row)

                if done:
                    if reward == 0.5:
                        gs.winCount('draw')
                    elif agent == ma:
                        gs.winCount('black')
                    elif agent == qa:
                        gs.winCount('white')
                    gs.print_history()
