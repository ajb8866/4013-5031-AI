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
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        scaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        if action == 'Stop': return -float("inf")
        successor = currentGameState.generatePacmanSuccessor(action)
        newPos = successor.getPacmanPosition()
        ghostStates = successor.getGhostStates()
        scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
        ghosts = len(ghostStates)

        for i in range(ghosts):
            if ghostStates[i].getPosition() == newPos and scaredTimes[i] == 0:
                return -float("inf")

        return 1000 - min([manhattanDistance(i, newPos) for i in currentGameState.getFood().asList()])


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
        return max([(self.minimax(gameState.generateSuccessor(0, action), depth - 1, 1, agents), action) for action in
                    gameState.getLegalActions(0)])[1]


    def minimax(self, gameState, depth, index, agents):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if not index:
            return max(
                [self.minimax(gameState.generateSuccessor(index, i), depth - 1, (index + 1) % agents, agents) for i in
                 gameState.getLegalActions(index)])
        else:
            return min(
                [self.minimax(gameState.generateSuccessor(index, i), depth - 1, (index + 1) % agents, agents) for i in
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
        m = -float('inf')
        depth = agents * self.depth
        for i in gameState.getLegalActions(0):
            v = self.minimaxPrun(gameState.generateSuccessor(0, i), depth - 1, 1, agents, alpha, beta)
            if v > m:
                action = i
                m = v
            if v > beta:
                return action
            alpha = max(v, alpha)
        return action

    # alpha and beta will store their current values in order to prune our tree
    def minimaxPrun(self, gameState, depth, index, agents, alpha, beta):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if not index:
            v = -float('inf')
            for i in gameState.getLegalActions(index):
                v = max(v,
                        self.minimaxPrun(gameState.generateSuccessor(index, i), depth - 1, (index + 1) % agents, agents,
                                         alpha, beta))
                if v > beta:
                    return v
                # If it's not, our current alpha will change
                alpha = max(v, alpha)
            return v
        else:
            # v will store our current minimum
            # When we get to a value smaller than our alpha, we will prune the rest of nodes
            v = float('inf')
            for i in gameState.getLegalActions(index):
                v = min(v,
                        self.minimaxPrun(gameState.generateSuccessor(index, i), depth - 1, (index + 1) % agents, agents,
                                         alpha, beta))
                if v < alpha:
                    return v
                # If it's not, our current beta will change
                beta = min(v, beta)
            return v


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
        max([(self.expectiminimax(gameState.generateSuccessor(0, action), depth - 1, 1, agents), action) for action in
             gameState.getLegalActions(0)])[1]

    def expectiminimax(self, gameState, depth, index, agents):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        newAgent = (index + 1) % agents
        if not index:
            return max([self.expectiminimax(gameState.generateSuccessor(index, i), depth - 1, newAgent, agents) for i in
                        gameState.getLegalActions(index)])
        else:
            beta = sum([self.expectiminimax(gameState.generateSuccessor(index, i), depth - 1, newAgent, agents) for i in
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
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # food will evaluate where all the pellets are
    food = sum([manhattanDistance(food, newPos) for food in newFood])

    # ghost will evaluate if the ghost is close to Pacman and if Pacman can or not eat it
    ghost = 0
    # Check every ghost
    for i in range(len(ghostStates)):
        d = manhattanDistance(newPos, ghostStates[i].getPosition())
        # To give more credit to distances, we will work with 10
        # If the ghost is scared, we want to go after it
        if scaredTimes[0] > 0:
            ghost += 10.0
        # If the ghost is not scared and we are too close to it, we have to run away from it
        if scaredTimes[i] == 0 and d < 1:
            ghost -= 1. / (1 - d)
        # If we can get to it before it stops being scared, we will try to eat it
        elif scaredTimes[i] < d:
            ghost += 1. / d
    # Finally, we join all this values
    return 1. / (1 + food * len(newFood)) + 10 * ghost + currentGameState.getScore()


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
        max([(self.boundedintelligencemaxagent(gameState.generateSuccessor(0, action), depth - 1, 1, agents), action)
             for action in gameState.getLegalActions(0)])[1]

    def boundedintelligencemaxagent(self, gameState, depth, index, agents):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        newAgent = (index + 1) % agents
        if not index:
            return max(
                [self.boundedintelligencemaxagent(gameState.generateSuccessor(index, i), depth - 1, newAgent, agents)
                 for i in gameState.getLegalActions(index)])
        else:
            l = len(gameState.getLegalActions(index))
            ghost = [
                self.boundedintelligencemaxagent(gameState.generateSuccessor(index, i), depth - 1, newAgent, agents) for
                i in gameState.getLegalActions(index)]
            # Gets the minimal value
            vm = min(ghost)
            beta = 3 * vm
            for i in range(1, l):
                beta += ghost[i] * (ghost[i] != vm)
            return float(beta / (l + 2.))
