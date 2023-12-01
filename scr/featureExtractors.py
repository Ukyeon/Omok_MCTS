import util
from StonePiece import StonePiece

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def __init__(self):
        self.initialize_features()

    def initialize_features(self):
        self.features = util.Counter()
        self.features["bias"] = 1.0
        self.features["black-1"] = 0.0
        self.features["black-2"] = 0.0
        self.features["black-3"] = 0.0
        self.features["black-4"] = 0.0
        self.features["black-5"] = 0.0
        self.features["white-1"] = 0.0
        self.features["white-2"] = 0.0
        self.features["white-3"] = 0.0
        self.features["white-4"] = 0.0
        self.features["white-5"] = 0.0

    def resetFeatures(self):
        self.initialize_features()

    def getFeatures(self, state, action):
        features = util.Counter()
        features["bias"] = 1.0
        # features["totalCount"] = state.getPlayerTurn()

        ori_x, ori_y = action
        color = state.getColor(ori_x, ori_y)
        mat = state.matrix.copy()
        stone = ['black', 'white']

        if color is None:
            if state.getPlayerTurn() % 2 == 1:  # Even : black / Odd : white
                mat[ori_x][ori_y] = StonePiece(ori_x, ori_y, 'white')
            else:
                mat[ori_x][ori_y] = StonePiece(ori_x, ori_y, 'black')

        for r in range(state.dimension):
            for col in stone:
                tensor = self.calculate_tensor(mat[r], state.dimension, col)
                key = self.getKey(col, r)
                features[key] = tensor * 0.1

        if color is None:
            if state.getPlayerTurn() % 2 == 1:  # Even : black / Odd : white
                mat[ori_x][ori_y] = 0
            else:
                mat[ori_x][ori_y] = 0

        return features


    def getFeatures2(self, state, action):
        ori_x, ori_y = action
        n_dx = [0, 0, 1, -1, -1, 1, -1, 1]
        n_dy = [-1, 1, -1, 1, 0, 0, -1, 1]
        turn = -1
        color = state.getColor(ori_x, ori_y)

        if color == None:
            turn = state.getPlayerTurn()  # Even : black / Odd : white
            features_tmp = self.features.copy()
            if turn % 2 == 1:
                color = 'white'
            else:
                color = 'block'

        connected = 0

        for direct in range(0, len(n_dy), 2):
            cnt = 1
            for next in range(direct, direct + 2):
                dx = ori_x
                dy = ori_y
                while True:
                    dx += n_dx[next]
                    dy += n_dy[next]
                    if color == state.getColor(dx, dy):
                        cnt += 1
                        connected += 1
                    else:
                        break

            if cnt > 1:
                key = self.getKey(color, cnt)
                if turn == -1:
                    self.features[key] += 1
                else:
                    features_tmp[key] += 1

        if connected == 0:
            key = self.getKey(color, 1)
            if turn == -1:
                self.features[key] += 1
            else:
                features_tmp[key] += 1

        return self.features if turn == -1 else features_tmp

    def calculate_tensor(self, matrix, dim, color):
        ans = 0
        place = 1
        for i in range(dim):
            if matrix[i] != 0 and matrix[i].get_color() == color:
                ans += place
            place = place * 2
        return ans

    def checkWindow(self, window):
        cnt = 1
        for i in range(1, 5):
            if window[i] == 'F':
                continue
            elif window[i] == window[0]:
                cnt += 1
            else:
                cnt = 0
                break 
        return cnt

    def getKey(self, color, num):
        if color == 'black':
            key = "black-" + str(num)
        else:
            key = "white-" + str(num)

        #print(key)
        return key

    def scoreEvaluationFunction(self, currentGameState):
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
                tmp[y][x] = currentGameState.getColor(y, x)

                if currentGameState.getColor(y, x) == None:
                    sum = 0

                    for next in range(0, len(n_dy)):
                        cnt = 0
                        dx, dy = x + n_dx[next], y + n_dy[next]
                        if currentGameState.getColor(dy, dx) == 'black':
                            cnt = 1.0
                            while True:
                                dx, dy = dx + n_dx[next], dy + n_dy[next]
                                
                                if currentGameState.checkLocation(dx, dy, currentGameState.dimension) == False or currentGameState.getColor(dy, dx) == None:
                                    break
                                elif currentGameState.getColor(dy, dx) == 'black':
                                    cnt *= 2
                                else:
                                    cnt /= 2
                                    break
                        elif currentGameState.getColor(dy, dx) == 'white':
                            cnt = -0.99
                            while True:
                                dx, dy = dx + n_dx[next], dy + n_dy[next]
                                
                                if currentGameState.checkLocation(dx, dy, currentGameState.dimension) == False or currentGameState.getColor(dy, dx) == None:
                                    break
                                elif currentGameState.getColor(dy, dx) == 'white':
                                    cnt *= 2
                                else:
                                    cnt /= 2
                                    break
                        sum += cnt
                    tmp[y][x] = sum
                    ans += sum

        return ans


class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features
