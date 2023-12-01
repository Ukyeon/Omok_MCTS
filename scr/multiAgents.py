# multiAgents.py
# --------------
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


from util import manhattanDistance
#import gameState
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print('scores: ', scores)
        # print('bestScore', bestScore)
        # print('bestIndices!!! ', bestIndices)
        # print('chosenIndex!', chosenIndex)
        # print('getAction====', legalMoves[chosenIndex])

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #print('cur pos: ', currentGameState.getPacmanPosition())
        # print('New Pos: ', newPos, action)
        # print('New Food!', newFood.asList())   
        # print('newScaredTimes!', newScaredTimes)
        #print('newGhostStates!!! ', newGhostStates) #newGhostStates[0], newGhostStates[1]) 

        frontier = util.Queue() # BFS 
        frontier.push( successorGameState )
        explored = []
        score = 0

        while True:
            if frontier.isEmpty(): 
                #print('Empth score : ', score)
                return score
            else:
                state = frontier.pop() # first node in frontier
                pacman_pos = state.getPacmanPosition()
                explored.append(pacman_pos) # Add curr_node state to explored

                distancesToFood = [manhattanDistance(p, pacman_pos) for p in newFood.asList()]
                distancesToGhost = [manhattanDistance(p.configuration.pos, pacman_pos) for p in newGhostStates]
                
                food_pos = newFood.asList() # current`s latest state!
                ghost_pos = [p.configuration.pos for p in state.getGhostStates()]
                score = score - min(distancesToFood) + min(distancesToGhost)

                if (min(distancesToGhost) < 2):
                    #print('distancesToGhost : ', score)
                    return float("-inf")
                if state.isWin() : # End with Win
                    if action == Directions.STOP :
                        return float("-inf") # Worst case
                    else :
                        return score # Win game, Return current score! 
                if state.isLose() :  # End with Lose
                    #print('End with Lose : ', score)
                    return float("-inf")
                if pacman_pos in food_pos : # Find food
                    #print('End with food : ', score)
                    return score + 100
                if pacman_pos in ghost_pos : # Find ghost
                    if newScaredTimes[ghost_pos.index(pacman_pos)] > 0 :
                        return score + 1000
                    else :
                        return score - 500
                if score < 50:
                    return score - sum(distancesToFood)
                    
                legalMoves = state.getLegalActions()
                for succ_act in legalMoves: # iterate all successors
                    successors = state.generatePacmanSuccessor(succ_act)
                    if  (successors.getPacmanPosition() not in explored) : 
                        frontier.push( successors )


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Omok GUI.

    This evaluation function is meant for use with adversarial search agents
    """
    tmp = [[0 for x in range(currentGameState.dimension)] for y in range(currentGameState.dimension)] #currentGameState.__str__()
    n_dx = [0, 0, 1, -1, -1, 1, -1, 1]
    n_dy = [-1, 1, -1, 1, 0, 0, -1, 1]

    ans = 0

    for y in range(0, currentGameState.dimension):
        for x in range(0, currentGameState.dimension):
            #tmp[y][x] = currentGameState.getColor(y, x)

            if currentGameState.getColor(x, y) == None:
                sum = 0

                for next in range(0, len(n_dy)):
                    cnt = 0
                    dx, dy = x + n_dx[next], y + n_dy[next]
                    if currentGameState.getColor(dx, dy) == 'black':
                        cnt = 1.0
                        while True:
                            dx, dy = dx + n_dx[next], dy + n_dy[next]
                            
                            if currentGameState.checkLocation(dx, dy, currentGameState.dimension) == False or currentGameState.getColor(dx, dy) == None:
                                break
                            elif currentGameState.getColor(dx, dy) == 'black':
                                cnt *= 2.9
                            else:
                                cnt /= 1.3
                                break
                    elif currentGameState.getColor(dx, dy) == 'white':
                        cnt = -0.99
                        while True:
                            dx, dy = dx + n_dx[next], dy + n_dy[next]
                            
                            if currentGameState.checkLocation(dx, dy, currentGameState.dimension) == False or currentGameState.getColor(dx, dy) == None:
                                break
                            elif currentGameState.getColor(dx, dy) == 'white':
                                cnt *= 2.7
                            else:
                                cnt /= 1.2
                                break
                    sum += cnt
                #tmp[y][x] = sum
                ans += sum

    #print(tmp)

    return ans

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', **args):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = args['depth']
        self.dimension = args['dimension']

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getLegalActions(self, state):
        matrix = state.matrix
        #print(matrix)
        actions = [(x, y) for x in range(self.dimension) for y in range(self.dimension) if matrix[x][y] == 0]
        #print("actions:", actions)

        return actions

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """          
        "*** YOUR CODE HERE ***"
        def maxValue(depth, gameState, alpha, beta, action):
            if depth == self.depth or self.is_gameOver(gameState, action):
                return self.evaluationFunction(gameState)

            value = float("-inf")
            legalMoves = self.getLegalActions(gameState) # My next action

            for action in legalMoves:
                value = max(value, minValue(depth + 1, gameState.generateSuccessor(action), alpha, beta, action))
                if value > beta : return value
                alpha = max(alpha, value)
            return value

        def minValue(depth, gameState, alpha, beta, action):
            if depth == self.depth or self.is_gameOver(gameState, action):
                return self.evaluationFunction(gameState)

            #print('alpha==:', alpha, 'NumAgents:', gameState.getNumAgents())
            value = float("+inf")
            legalMoves = self.getLegalActions(gameState) # My next action

            for action in legalMoves:
                value = min(value, maxValue(depth + 1, gameState.generateSuccessor(action), alpha, beta, action))

                if value < alpha: 
                    #print('agentIndex==:', agentIndex, 'NumAgents:', gameState.getNumAgents())
                    return value

                beta = min(beta, value)

            return value

        if gameState.getPlayerTurn() == 0:
            return gameState.getCenter()

        legalMoves = self.getLegalActions(gameState)
        alpha = float("-inf")
        beta = float("+inf")
        value = float("-inf")
        ans = []

        for action in legalMoves:
            tmp = minValue(0, gameState.generateSuccessor(action), alpha, beta, action) # start depth == 0, Myagent = 1
            #print(action, tmp)

            if tmp > value:
                value = tmp
                ans.clear()
                ans.append(action)
            elif tmp == value:
                ans.append(action)

            alpha = max(alpha, value)
        
        #print('ANS:', ans, 'NumAgents:', gameState.getPlayerTurn())
        return random.choice(ans)

    def is_gameOver(self, gameState, action):
        if gameState.getPlayerTurn() >= self.dimension * self.dimension: # Draw
            return True

        ori_x, ori_y = action
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
                    
                    if gameState.checkLocation(x, y, self.dimension) == False or gameState.getColor(x, y) == None:
                        break
                    elif gameState.getColor(x, y) == 'black':  # MINMAX == black
                        cnt += 1
                    else:
                        break
                    
            if cnt >= 5:
                #print("player " + stone + " wins!!")
                return True
        return False


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
            
        def maxValue(depth, gameState):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            value = float("-inf")
            legalMoves = gameState.getLegalActions(0) # My next action

            for action in legalMoves:
                value = max(value, expValue(depth, gameState.generateSuccessor(0, action), 1))
            return value

        def expValue(depth, gameState, agentIndex):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            value = 0
            legalMoves = gameState.getLegalActions(agentIndex) # Enemies next action

            for action in legalMoves:
                if agentIndex == gameState.getNumAgents() - 1: # Increase depth only here due to gameState changes from this.
                    value += maxValue(depth + 1, gameState.generateSuccessor(agentIndex, action))
                else: # Increase agentIndex only here due to gameState isn`t changed yet.
                    value += expValue(depth, gameState.generateSuccessor(agentIndex, action), agentIndex + 1)
                    #print('agentIndex==:', agentIndex, 'NumAgents:', gameState.getNumAgents()) 
            #weights = [0.5] # All ghosts should be modeled as choosing uniformly at random
            return value / len(legalMoves)
        
        legalMoves = gameState.getLegalActions(0)
        v = float("-inf")
        ans = Directions.STOP

        for action in legalMoves:
            tmp = expValue(0, gameState.generateSuccessor(0, action), 1) # start depth == 0, Myagent = 1
            if tmp > v:
                v = tmp
                ans = action

        #print('ANS:', ans, 'NumAgents:', gameState.getNumAgents())
        return ans
        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Use BFS to figure out the shortest distance to a food.
                And then bagically return min(distancesToGhost) - sum(distancesToFood) - the shortest distance
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin() : # End with Win
        return float("+inf") # Worst case
    if currentGameState.isLose() :  # End with Lose
        return float("-inf")

    pacman_pos = currentGameState.getPacmanPosition()
    food_pos = currentGameState.getFood().asList() # current`s latest state!
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    walls = currentGameState.getWalls().asList()
    
    distancesToFood = [manhattanDistance(p, pacman_pos) for p in food_pos]
    distancesToGhost = [manhattanDistance(p.configuration.pos, pacman_pos) for p in newGhostStates]

    frontier = util.Queue() # BFS 
    frontier.push( (pacman_pos, 0) )
    explored = []

    while True:
        if frontier.isEmpty(): 
            break
        else:
            pacman_pos, cost = frontier.pop() # first node in frontier
            score = cost
            explored.append((pacman_pos)) # Add curr_node state to explored

            if pacman_pos in food_pos: #pacman_find food!
                break

            for succX, succY in [(0, 1) , (0 , -1) , (-1 , 0) , (1, 0)]:
                nextX = pacman_pos[0] + succX
                nextY = pacman_pos[1] + succY
                if (nextX, nextY) not in explored and (nextX, nextY) not in walls:
                    frontier.push( ((nextX, nextY), cost + 1) )

    if sum(newScaredTimes) == 0 :
        ret = min(distancesToGhost) - sum(distancesToFood) - score
    else:
        ret = sum(newScaredTimes) - sum(distancesToFood) - score

    ret -= 300 * len(distancesToFood)  # Some constant value for auto grader..
    #print(ret)
    return ret

# Abbreviation
better = betterEvaluationFunction
