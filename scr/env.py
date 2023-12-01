from enum import IntEnum

import random

from gym.utils import seeding



class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

class Action_diag(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    NONE_MOVE = 8

def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]


def actions_to_dxdy_kingsMove(action: Action_diag) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action_diag.LEFT: (-1, 0),
        Action_diag.DOWN: (0, -1),
        Action_diag.RIGHT: (1, 0),
        Action_diag.UP: (0, 1),
        Action_diag.UP_LEFT: (-1, 1),
        Action_diag.UP_RIGHT: (1, 1),
        Action_diag.DOWN_LEFT: (-1, -1),
        Action_diag.DOWN_RIGHT: (1, -1),
        Action_diag.NONE_MOVE: (0, 0),
    }
    return mapping[action]


class DQNEnv(Env):
    def __init__(self):
        """DQN for omok grid world gym environment
        """

        self.state = GameState(dimension=dimension)  # Initialize the game state
        # Grid dimensions (x, y)
        self.rows = 10
        self.cols = 7

        # Wind
        # either a dict (keys would be states) or multidimensional array (states correspond to indices)
        self.wind = {(i, j): 2 if i in (6, 7) else 1 for i in range(3, 8) for j in range(0, 7)}

        self.diagonal_moves = diagonal_moves
        if diagonal_moves:
            self.action_space = spaces.Discrete(len(Action_diag))
        else:
            self.action_space = spaces.Discrete(len(Action))

        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

        # Set start_pos and goal_pos
        self.start_pos = (0, 3)
        self.goal_pos = (7, 3)
        self.agent_pos = None


    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = 0.0
        else:
            done = False
            reward = -1.0

        if self.diagonal_moves:
            x, y = actions_to_dxdy_kingsMove(action)
        else:
            x, y = actions_to_dxdy(action)
        next_pos = (self.agent_pos[0] + x, self.agent_pos[1] + y)
        #print("before wind: ", next_pos)
        blown = self.wind[self.agent_pos] if self.agent_pos in self.wind else 0
        if self.stochastic_wind:
            rand = random.random()
            if rand < 1/3:
                blown -= 1
            elif rand < 2/3:
                blown += 1

        next_pos = (self.agent_pos[0] + x, max(self.agent_pos[1] + y - blown, 0))  # blown by wind
        #print("next wind: ", next_pos)

        # If the next position is a wall or out of bounds, stay at current position
        if 0 > next_pos[0] or self.rows < next_pos[0] or 0 > next_pos[1] or self.cols < next_pos[1]:
            next_pos = self.agent_pos

        # Set self.agent_pos
        self.agent_pos = next_pos

        return self.agent_pos, reward, done, {}



