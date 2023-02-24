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
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        total = 0

        #using built in score
        total += scoreEvaluationFunction(successorGameState)

        #food positions = 1/distance to closest food
        distances = []
        for food in newFood.asList():
            distances.append(manhattanDistance(food, newPos))
        if len(distances)>0:
            total -= 1.5*((sum(distances)/len(distances)))
            total -= 0.5 * min(distances)
            total -= 12 * len(newFood.asList())
        #ghost positions = distance to closest ghost
        distances2 = []
        for ghost in newGhostStates:
            distances2.append(manhattanDistance(ghost.getPosition(), newPos))
        if len(distances2) > 0:
            total += min(distances2)
        
        return total

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        return self.minimaxDFS(gameState, gameState.getNumAgents(), 0, 1)[1]

        #util.raiseNotDefined()

    def minimaxDFS(self, gameState: GameState, numAgents: int, agentConsidered: int, depth: int):
        if depth == self.depth + 1:
            return (self.evaluationFunction(gameState), None)
        if (gameState.isWin() or gameState.isLose()):
            return (self.evaluationFunction(gameState), None)

        actions = gameState.getLegalActions(agentConsidered)
        minOrMax = None
        score = 0
        if agentConsidered == 0:
            minOrMax = max
            score = -2000000000
        else:
            minOrMax = min
            score = 2000000000

        returnedAction = None

        for action in actions:
            newState = gameState.generateSuccessor(agentConsidered, action)

            if agentConsidered + 1 >= numAgents:
                temp = self.minimaxDFS(newState, numAgents, 0, depth + 1)
                score = minOrMax(temp[0], score)
                if score == temp[0]:
                    returnedAction = action
            else:
                temp = self.minimaxDFS(newState, numAgents, agentConsidered + 1, depth)
                score = minOrMax(temp[0], score)
                if score == temp[0]:
                    returnedAction = action

        return (score,returnedAction)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBetaDFS(gameState, gameState.getNumAgents(), 0, 1, -2000000000,2000000000)[1]
        #util.raiseNotDefined()

    
    def alphaBetaDFS(self, gameState: GameState, numAgents: int, agentConsidered: int, depth: int, alpha: int, beta: int):
        if depth == self.depth + 1:
            return (self.evaluationFunction(gameState), None)
        if (gameState.isWin() or gameState.isLose()):
            return (self.evaluationFunction(gameState), None)

        actions = gameState.getLegalActions(agentConsidered)
        minOrMax = None
        score = 0
        if agentConsidered == 0:
            minOrMax = max
            score = -2000000000
        else:
            minOrMax = min
            score = 2000000000

        returnedAction = None

        for action in actions:
            newState = gameState.generateSuccessor(agentConsidered, action)

            if agentConsidered + 1 >= numAgents:
                temp = self.alphaBetaDFS(newState, numAgents, 0, depth + 1, alpha, beta)
                prev = score
                score = minOrMax(temp[0], score)
                if score == temp[0]:
                    returnedAction = action
            else:
                temp = self.alphaBetaDFS(newState, numAgents, agentConsidered + 1, depth, alpha, beta)
                prev = score
                score = minOrMax(temp[0], score)
                if score == temp[0]:
                    returnedAction = action

            if(agentConsidered == 0 and score > beta):
                return (score, returnedAction, alpha, beta)
            elif (agentConsidered != 0 and score< alpha):
                return (score, returnedAction, alpha, beta)
            elif agentConsidered == 0:
                alpha = max(score, alpha)
            elif agentConsidered != 0:
                beta = min(score, beta)

        return (score,returnedAction)
    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimaxDFS(gameState, gameState.getNumAgents(), 0, 1)[1]

    def expectimaxDFS(self, gameState: GameState, numAgents: int, agentConsidered: int, depth: int):
        if depth == self.depth + 1:
            return (self.evaluationFunction(gameState), None)
        if (gameState.isWin() or gameState.isLose()):
            return (self.evaluationFunction(gameState), None)

        actions = gameState.getLegalActions(agentConsidered)
        score = 0
        if agentConsidered == 0:
            score = -2000000000


        returnedAction = None

        if (agentConsidered == 0):        
            for action in actions:
                newState = gameState.generateSuccessor(agentConsidered, action)
                newDepth = depth
                if (numAgents == 1):
                    newDepth += 1
                temp = self.expectimaxDFS(newState, numAgents, (agentConsidered + 1) % numAgents, newDepth)
                score = max(temp[0], score)
                if score == temp[0]:
                    returnedAction = action
        else:
            for action in actions:
                newState = gameState.generateSuccessor(agentConsidered, action)
                newDepth = depth
                if (numAgents - 1 == agentConsidered):
                    newDepth += 1
                temp = self.expectimaxDFS(newState, numAgents, (agentConsidered + 1) % numAgents, newDepth)
                score += temp[0]
            
        if (agentConsidered == 0):
            return (score,returnedAction)
        else:
            return (score / len(actions), random.choice(actions))

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    Our evaluation function is a function of score, average distance to food, distance 
    to closest food, number of food left, distance to ghosts, adjacency to ghosts, 
    adjacency to power pellet, and scared time > min(distance to ghost).
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newPowerPellets = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    total = 0

    #using built in score
    total += scoreEvaluationFunction(currentGameState)

    #food positions = 1/distance to closest food
    distances = []
    for food in newFood.asList():
        distances.append(manhattanDistance(food, newPos))

    if len(distances)>0:
        total -= 1.5*((sum(distances)/len(distances)))
        total -= 0.5 * min(distances)
        total -= 12 * len(newFood.asList())

    #ghost positions = distance to closest ghost
    scaredTime = sum(newScaredTimes)
    distances2 = []
    for ghost in newGhostStates:
        distances2.append(manhattanDistance(ghost.getPosition(), newPos))
    if scaredTime == 0:
        if len(distances2) > 0:
            total += min(distances2)*2
        if min(distances2) == 0:
            total + 100
    #ghost scared times
    
    if (scaredTime > 7 and scaredTime >= min(distances2)):
        total += 10000000
    if scaredTime >= min(distances2) and scaredTime>1:
        total -= min(distances2)*1000
    
    return total

# Abbreviation
better = betterEvaluationFunction
