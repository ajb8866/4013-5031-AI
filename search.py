# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


#########################
# Angel Bergantiños Yeste#
#########################


from util import manhattanDistance
from game import Directions
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIdx = [index for index in range(len(scores)) if scores[index] == bestScore]
        # pick an action that is among the best
        chosen = random.choice(bestIdx)

        "Add more of your code here if you want to"

        return legalMoves[chosen]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (updatedPos).
        scaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # If Packman does not move
        if action == 'Stop': return -float("inf")
        successor = currentGameState.generatePacmanSuccessor(action)

        # If Pacman moves
        # update Pacman posistion
        updatedPos = successor.getPacmanPosition()
        # update ghost states
        ghostStates = successor.getGhostStates()
        # update how long the ghost are scared (if at all)
        scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
        ghosts = len(ghostStates)

        # If ghost can kill, dont move in direction of ghost
        for i in range(ghosts):
            if ghostStates[i].getPosition() == updatedPos and scaredTimes[i] == 0:
                return -float("inf")

        #If no killer ghost, move to closest food
        return 1000 - min([manhattanDistance(i, updatedPos) for i in currentGameState.getFood().asList()])


######################
###End ReflexAgent####
######################

def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


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
        """
        agents = gameState.getNumAgents()
        depth = agents * self.depth
        return max([(self.MiniMax(gameState.generateSuccessor(0, action), depth - 1, 1, agents), action) for action in
                    gameState.getLegalActions(0)])[1]
    # Calculate minmax with the current gamestate, depth of current graph, current agents, and agent amounts
    def MiniMax(self, gameState, depth, index, agents):
        # Evaluate current game at (if Pacman won, lost, or the is no more depth in the graph)
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        # Pacman agent is 0 (so not index), look for max
        if not index:
            return max(
                [self.MiniMax(gameState.generateSuccessor(index, i), depth - 1, (index + 1) % agents, agents) for i in
                 gameState.getLegalActions(index)])
        else:
            # If a ghost (at index), look for mini
            return min(
                [self.MiniMax(gameState.generateSuccessor(index, i), depth - 1, (index + 1) % agents, agents) for i in
                 gameState.getLegalActions(index)])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        agents = gameState.getNumAgents()
        alpha = -float('inf')
        beta = float('inf')
        temp = -float('inf')
        depth = agents * self.depth
        for i in gameState.getLegalActions(0):
            value = self.pruning(gameState.generateSuccessor(0, i), depth - 1, 1, agents, alpha, beta)
            if value > temp:
                action = i
                temp = value
            if value > beta:
                return action
            alpha = max(value, alpha)
        return action

    # Acts as the pruning function for alpha and beta
    # The two will store their current values for pruning
    def pruning(self, gameState, depth, index, agents, alpha, beta):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if not index:
            value = -float('inf')
            for legal in gameState.getLegalActions(index):
                value = max(value,
                        self.pruning(gameState.generateSuccessor(index, legal), depth - 1, (index + 1) % agents, agents,
                                         alpha, beta))
                # new value is smaller then old return new smaller value
                if value > beta:
                    return value
                # new value is not smaller then old, check if it can be alpha
                alpha = max(value, alpha)
            return value
        else:
            # Prune the rest of nodes because value is smaller then alpha
            value = float('inf')
            for legal in gameState.getLegalActions(index):
                value = min(value,
                        self.pruning(gameState.generateSuccessor(index, legal), depth - 1, (index + 1) % agents, agents,
                                         alpha, beta))
                # new value is larger then old return new larger value

                if value < alpha:
                    return value
                # new value is smaller then old, check if it can be beta
                beta = min(value, beta)
            return value


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
        agents = gameState.getNumAgents()
        depth = agents * self.depth
        return \
            max([(self.ExpectMiniMax(gameState.generateSuccessor(0, action), depth - 1, 1, agents), action) for action
                 in
                 gameState.getLegalActions(0)])[1]

    # Works like minimax with probabilities added
    def ExpectMiniMax(self, gameState, depth, index, agents):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        newAgent = (index + 1) % agents
        if not index:
            return max([self.ExpectMiniMax(gameState.generateSuccessor(index, i), depth - 1, newAgent, agents) for i in
                        gameState.getLegalActions(index)])
        else:
            # The difference between the two minimax functions  is that we will use as a mean for all solutions
            beta = sum([self.ExpectMiniMax(gameState.generateSuccessor(index, i), depth - 1, newAgent, agents) for i in
                        gameState.getLegalActions(index)])
            return float(beta / len(gameState.getLegalActions(index)))


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: We will take in account some factors:
        -Distances to the food
        -Is the ghost scared? How far is the ghost from Pacman?
    """
    updatedPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Food is where dots are
    food = sum([manhattanDistance(food, updatedPos) for food in newFood])

    # Checks posistion of ghost and their edibility for all ghost
    ghostValue = 0
    for ghost in range(len(ghostStates)):
        dist = manhattanDistance(updatedPos, ghostStates[ghost].getPosition())
        # If the ghost is scared Pacman tries to eat the ghost
        if scaredTimes[0] > 0:
            ghostValue += 10.0
        # If the ghost is not scared Pacman runs aways
        if scaredTimes[ghost] == 0 and dist < 1:
            ghostValue -= 1. / (1 - dist)
        # If we can get to it before the scared timer runs out, Pacman will tries to eat it
        elif scaredTimes[ghost] < dist:
            ghostValue += 1. / dist

    return 1. / (1 + food * len(newFood)) + 10 * ghostValue + currentGameState.getScore()


# Abbreviation
better = betterEvaluationFunction


class BoundedIntelligenceMaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        agents = gameState.getNumAgents()
        depth = agents * self.depth
        return \
            max([(
                 self.boundedintelligencemaxagent(gameState.generateSuccessor(0, action), depth - 1, 1, agents), action)
                 for action in gameState.getLegalActions(0)])[1]

    def boundedintelligencemaxagent(self, gameState, depth, index, agents):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        newAgent = (index + 1) % agents
        if not index:
            return max(
                [self.boundedintelligencemaxagent(gameState.generateSuccessor(index, legal), depth - 1, newAgent, agents)
                 for legal in gameState.getLegalActions(index)])
        else:
            # The difference between minimax and boundedintelligencemaxagent is used as the mean of possible outputs
            legalLength = len(gameState.getLegalActions(index))
            ghost = [
                self.boundedintelligencemaxagent(gameState.generateSuccessor(index, legal), depth - 1, newAgent, agents) for
                legal in gameState.getLegalActions(index)]
            # finds min value
            valueMin = min(ghost)
            beta = 3 * valueMin
            for legal in range(1, legalLength):
                # if min is found do not add again
                beta += ghost[legal] * (ghost[legal] != valueMin)
            return float(beta / (legalLength + 2.))
