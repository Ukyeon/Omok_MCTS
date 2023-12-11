from StonePiece import StonePiece
from typing import Tuple, Optional, List
import gym
from gym import Env, spaces
from gym.envs.registration import register
import numpy as np

board_size = 15
MATRIX = [[0 for x in range(board_size)] for y in range(board_size)]

# for i in range(board_size):
#     for j in range(board_size):
#         MATRIX[i][j] = "empty"

bC = StonePiece(1, 0, 'black')
wC = StonePiece(1, 0, 'white')
#MATRIX[0][0] = bC
#MATRIX[2][3] = wC

class GameState(Env):
    def __init__(self, **args):
        self.highlightSwitch = 0
        self.dimension = args['dimension']
        self.matrix = MATRIX  # Grid(self.dimension, self.dimension)
        self.player_turn = 0  # Even : black / Odd : white
        self.win_history = [0, 0, 0]  # black, white, draw
        self.action_space = spaces.Discrete(self.dimension*self.dimension)

        # Assuming self.matrix represents row states as binary values for stones' positions
        # Define the minimum and maximum values for each row
        min_row_values = np.zeros(self.dimension, dtype=np.float32)
        max_row_values = np.array([int('1' * self.dimension, 2)] * self.dimension, dtype=np.float32)

        self.observation_space = spaces.Box(min_row_values, max_row_values, dtype=np.float32)

    def __str__(self):
        width, height = self.dimension, self.dimension
        map = Grid(width, height)
        for row in range(width):
            for col in range(height):
                if self.getColor(row, col) == 'black':
                    map[col][row] = 'x'
                elif self.getColor(row, col) == 'white':
                    map[col][row] = 'o'
                else:
                    map[col][row] = 'F'
        return str(map) #+ ("\nScore: %d\n" % self.score)

    def flip_highlight_switch(self, value):
        self.highlightSwitch = value

    def get_highlight_switch(self):
        return self.highlightSwitch

    def is_invalid(self, x, y):
        return (x < 0 or x >= board_size or y < 0 or y >= board_size)

    def is_gameOver(self, x, y, stone):
        '''
        for i in range(h):
            for j in range(h):
                TODO: "GO through matrix and determine if win condition and player turn"
                if MATRIX[i][j] something
        '''

        if self.player_turn >= self.dimension * self.dimension: # Draw
            # self.winCount('draw')
            return True

        ori_x, ori_y = x, y
        n_dx = [0, 0, 1, -1, -1, 1, -1, 1]
        n_dy = [-1, 1, -1, 1, 0, 0, -1, 1]
        #print(stone, self.matrix[x][y].get_color())

        for direct in range(0, len(n_dy), 2):
            cnt = 1
            for next in range(direct, direct + 2):
                dx, dy = n_dx[next], n_dy[next]
                x, y = ori_x, ori_y
                
                while True:
                    x, y = x + dx, y + dy
                    #print(stone, x, y, dx, dy, cnt)
                    
                    if self.checkLocation(x, y, self.dimension) is False or self.matrix[x][y] == 0:
                        break
                    elif self.getColor(x, y) == stone:
                        cnt += 1
                    else:
                        break
                    
            if cnt >= 5:
                # self.winCount(stone)
                # print("player " + stone + " wins!!" self.win_history)
                return True
        return False

    def winCount(self, color):
        if color == 'black':
            self.win_history[0] += 1 
        elif color == 'white':
            self.win_history[1] += 1
        elif color == 'draw':
            self.win_history[2] += 1

    def checkLocation(self, row, col, dimension):
        if col < 0 or col >= dimension or row < 0 or row >= dimension:
            #print("Placing error!", col, row)
            return False
        return True

    def updatePlayerTurn(self, row, col):
        done = 0
        reward = 0
        if self.matrix[row][col] == 0:
            if self.player_turn % 2 == 0:
                self.matrix[row][col] = StonePiece(row, col, 'black')
            elif self.player_turn % 2 == 1:
                self.matrix[row][col] = StonePiece(row, col, 'white')
            self.player_turn += 1

            done, reward = self.getReward(col, row)

        return done, reward

    def getReward(self, col, row):
        done = 0
        if self.is_gameOver(col, row, self.getColor(col, row)) == True:
            if self.getPlayerTurn() >= self.dimension * self.dimension:  # Draw
                reward = 0.5
            elif self.getPlayerTurn() % 2 == 1:  # LOSE
                reward = -1
            else:
                reward = 1  # RL win

            done = 1
            # self.print_history()
            self.reset()
        else:
            reward = -0.01

        return done, reward


    def getPlayerTurn(self):
        return self.player_turn

    def getColor(self, row, col):
        if (self.checkLocation(row, col, self.dimension) is False) or (self.matrix[row][col] == 0):
            return None
        return self.matrix[row][col].get_color()

    def deepCopy(self):
        state = GameState(dimension=self.dimension)
        state.matrix = [x[:] for x in self.matrix]
        state.player_turn = self.player_turn
        return state

    def reset(self):
        self.highlightSwitch = 0
        self.player_turn = 0
        self.matrix = [[0 for x in range(board_size)] for y in range(board_size)]
        return np.zeros(self.dimension, dtype=np.float32)

    def reset_history(self):
        self.win_history = [0, 0, 0]

    def print_history(self):
        print(self.win_history)

    def generateSuccessor(self, action):
        successor = self.deepCopy()
        x, y = action
        done, reward = successor.updatePlayerTurn(x, y)
        return successor, reward, done

    def getCenter(self):
        return (int(self.dimension/2), int(self.dimension/2))

    def getLegalActions(self):
        return [(x, y) for x in range(self.dimension) for y in range(self.dimension) if self.matrix[x][y] == 0]

    def get_int(self, action):
        row, col = action
        return row * self.dimension + col

    def get_observation(self):
        # Convert '1' in self.matrix to the maximum binary value and others to minimum binary value
        observation = np.where(self.matrix == 1, int('1' * self.dimension, 2), 0)
        return observation.astype(np.float32)

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

        row, col = action
        self.updatePlayerTurn(row, col)
        # Check if goal was reached
        if self.is_gameOver(row, col, self.getColor(row, col)) == True:
            done = True
            reward = 0.0
        else:
            done = False
            reward = 0.0

        observation = self.get_observation()
        return observation, reward, done


class Grid:
    """
    A 2-dimensional array of objects backed by a list of lists.  Data is accessed
    via grid[x][y] where (x,y) are positions on a Omok map with x horizontal,
    y vertical and the origin (0,0) in the bottom left corner.

    The __str__ method constructs an output that is oriented like a Omok board.
    """

    def __init__(self, width, height, initialValue=False, bitRepresentation=None):
        if initialValue not in [False, True]:
            raise Exception('Grids can only contain booleans')
        self.CELLS_PER_INT = 30

        self.width = width
        self.height = height
        self.data = [[initialValue for y in range(height)] for x in range(width)]
        if bitRepresentation:
            self._unpackBits(bitRepresentation)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __str__(self):
        out = [[str(self.data[x][y])[0] for x in range(self.width)]
               for y in range(self.height)]
        # out.reverse()
        return '\n'.join([''.join(x) for x in out])

    def __eq__(self, other):
        if other == None:
            return False
        return self.data == other.data

    def __hash__(self):
        # return hash(str(self))
        base = 1
        h = 0
        for l in self.data:
            for i in l:
                if i:
                    h += base
                base *= 2
        return hash(h)

    def copy(self):
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def count(self, item=True):
        return sum([x.count(item) for x in self.data])

    def asList(self, key=True):
        list = []
        for x in range(self.width):
            for y in range(self.height):
                if self[x][y] == key:
                    list.append((x, y))
        return list

    def packBits(self):
        """
        Returns an efficient int list representation

        (width, height, bitPackedInts...)
        """
        bits = [self.width, self.height]
        currentInt = 0
        for i in range(self.height * self.width):
            bit = self.CELLS_PER_INT - (i % self.CELLS_PER_INT) - 1
            x, y = self._cellIndexToPosition(i)
            if self[x][y]:
                currentInt += 2 ** bit
            if (i + 1) % self.CELLS_PER_INT == 0:
                bits.append(currentInt)
                currentInt = 0
        bits.append(currentInt)
        return tuple(bits)

    def _cellIndexToPosition(self, index):
        x = index / self.height
        y = index % self.height
        return x, y

    def _unpackBits(self, bits):
        """
        Fills in data from a bit-level representation
        """
        cell = 0
        for packed in bits:
            for bit in self._unpackInt(packed, self.CELLS_PER_INT):
                if cell == self.width * self.height:
                    break
                x, y = self._cellIndexToPosition(cell)
                self[x][y] = bit
                cell += 1

    def _unpackInt(self, packed, size):
        bools = []
        if packed < 0:
            raise ValueError("must be a positive integer")
        for i in range(size):
            n = 2 ** (self.CELLS_PER_INT - i - 1)
            if packed >= n:
                bools.append(True)
                packed -= n
            else:
                bools.append(False)
        return bools


def register_env(**kwargs) -> None:
    """Register custom gym environment so that we can use `gym.make()`

    In your main file, call this function before using `gym.make()` to use the Four Rooms environment.
        register_env()
        env = gym.make('WindyGridWorld-v0')

    There are a couple of ways to create Gym environments of the different variants of Windy Grid World.
    1. Create separate classes for each env and register each env separately.
    2. Create one class that has flags for each variant and register each env separately.

        Example:
        (Original)     register(id="WindyGridWorld-v0", entry_point="env:WindyGridWorldEnv")
        (King's moves) register(id="WindyGridWorldKings-v0", entry_point="env:WindyGridWorldEnv", **kwargs)

        The kwargs will be passed to the entry_point class.

    3. Create one class that has flags for each variant and register env once. You can then call gym.make using kwargs.

        Example:
        (Original)     gym.make("WindyGridWorld-v0")
        (King's moves) gym.make("WindyGridWorld-v0", **kwargs)

        The kwargs will be passed to the __init__() function.

    Choose whichever method you like.
    """

    register(id="GameState-v0", entry_point="gameState:GameState", kwargs=kwargs)


def create_env(**kwargs):
    register_env()
    env = gym.make('GameState-v0', **kwargs)
    return env


