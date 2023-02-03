# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
from urllib3.util import queue

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    depth = util.Stack()
    depth.push((problem.getStartState(), []))
    hasVisited = list()

    while not depth.isEmpty():
        current, step = depth.pop()
        if current in hasVisited:
            continue
        if problem.isGoalState(current):
            return step
        hasVisited.append(current)
        for state, action, cost in problem.getSuccessors(current):
            depth.push((state, step + [action]))
    return []


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    breadth = util.Queue()
    breadth.push((problem.getStartState(), []))
    hasVisited = list()

    while not breadth.isEmpty():
        current, step = breadth.pop()
        if current in hasVisited:
            continue
        if problem.isGoalState(current):
            return step
        hasVisited.append(current)
        for state, action, cost in problem.getSuccessors(current):
            breadth.push((state, step + [action]))

    return []


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    uc = util.PriorityQueue()
    uc.push((problem.getStartState(), [], 0), 0)
    hasVisited = list()

    while not uc.isEmpty():
        current, step, existed = uc.pop()
        if current in hasVisited:
            continue
        if problem.isGoalState(current):
            return step
        hasVisited.append(current)
        for state, action, cost in problem.getSuccessors(current):
            uc.push((state, step + [action], existed + cost), existed + cost)
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    a = util.PriorityQueue()
    a.push((problem.getStartState(), [], 0), 0)
    hasVisited = dict()

    while not a.isEmpty():
        current, step, existed = a.pop()
        if current in hasVisited and hasVisited[current] <= existed:
            continue
        if problem.isGoalState(current):
            return step

        hasVisited[current] = existed
        for state, action, cost in problem.getSuccessors(current):
            a.push((state, step + [action], existed + cost), existed + cost + heuristic(state, problem))
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
