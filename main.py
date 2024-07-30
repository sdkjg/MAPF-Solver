import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import random
from queue import PriorityQueue as pq
import heapq as hq

# variables
mapLayout = []
width = 0
height = 0
mapPathSpace = []
mapWallSpaces = []
allPaths = []
agentColors = []
currentTime = 0
maxTime = 0
agentInfo = []
agents = 2
starts = []
ends = []
allCollisions = []
timeLimit = 5
test = [1, 4] #generate starting position for specific tests
AStarAllNodes = []

colors = ["spring", "summer", "autumn", "winter", "cool", "Wistia", "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "bwr",
          "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "tab10", "tab20", "tab20b"] #agent colors

class Node: #for A*
    def __init__(self):
        self.parent = None
        self.f = float('inf')
        self.g = float('inf')

def onMap(pos):
    return 0 <= pos[0] < width and 0 <= pos[1] < height

def startAndEnd(): #generate starting and ending locations
    # start = []
    # end = []
    # sPos = random.choice(mapPathSpace)
    # ePos = random.choice(mapPathSpace)
    # start.append(sPos[0])
    # start.append(sPos[1])
    # end.append(ePos[0])
    # end.append(ePos[1])
    # sPos = random.choice(test)  # small area testing
    # if sPos == 1:
    #     start = [1, 1]
    #     end = [4, 1]
    # else:
    #     start = [4, 1]
    #     end = [1, 1]
    sPos = random.choice(test)  # small area testing
    if sPos == 1:
        start = [1, 1]
        end = [4, 1]
    else:
        start = [4, 1]
        end = [1, 1]
    return start, end

def heuristicValue(pos, end):
    return abs(end[1] - pos[1]) + abs(end[0] - pos[0])

def readMap(mapFile): #read file information
    global width, height, mapLayout, mapPathSpace, mapWallSpaces
    with open(mapFile) as f:
        for line in f:
            line = line.strip()
            if line.startswith('width'):
                width = int(line.split()[1])
                continue
            elif line.startswith('height'):
                height = int(line.split()[1])
                continue
            elif line.startswith('map') or line.startswith('type'):
                continue
            mapLayout.append([line])
    mapArray = np.zeros((height, width), dtype=int)
    for y in range(height):
        for x in range(width):
            point = mapLayout[y][0][x]
            if point == '.':
                mapArray[y, x] = 1
                mapPathSpace.append([x, y])
            elif point == 'T':
                mapWallSpaces.append([x, y])
    return mapArray

def agentMap(agentPos, displayPos, clear): #create map of path for single agent
    addInfo = False
    if len(agentInfo) < agents:
        addInfo = True
    agentArray = np.zeros((height, width), dtype=int)
    if addInfo:
        agentInfo.append([])
    if clear:
        for pos in mapPathSpace:
            x, y = pos
            agentArray[y, x] = 1
    else:
        for pos in agentPos:
            x, y = pos
            if pos == displayPos:
                agentArray[y, x] = 1
            if addInfo:
                agentInfo[-1].append([x, y])
    return agentArray

def updateMap(val): #update map based on slider value for time
    global currentTime
    currentTime = int(timeSlider.val)
    agentArray = agentMap(None, None, True)
    agentArray = np.ma.masked_where(agentArray == 0, agentArray)
    axis.imshow(agentArray, cmap='binary', aspect='equal', origin='upper')
    for i in range(agents):
        agentPath = allPaths[i]
        color = agentColors[i]
        limit = min(len(agentPath), currentTime) + 1
        agentPos = agentPath[:limit][-1]
        agentArray = agentMap(agentPath, agentPos, False)
        agentArray = np.ma.masked_where(agentArray == 0, agentArray)
        axis.imshow(agentArray, cmap=color, aspect='equal', origin='upper')

def displayMap(mapFile): #starting command, generates agents, collisionless paths, and displays map and agents
    global timeSlider, axis, allPaths, maxTime
    maxTime = 0
    mapArray = readMap(mapFile)
    fig, axis = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    axes = plt.axes((0.15, 0.1, 0.7, 0.03))
    axis.imshow(mapArray, cmap='grey', aspect='equal', origin='upper')
    allAgents = []
    for i in range(agents):
        agentPath = aStar(None, None, [])
        allAgents.append(agentPath)
    finalAgentPaths = CBS(allAgents)
    allPaths = finalAgentPaths
    for path in allPaths:
        if len(path) > maxTime:
            maxTime = len(path) - 1
    for agentPath in allPaths:
        displayPos = agentPath[-1]
        agentArray = agentMap(agentPath, displayPos, False)
        agentArray = np.ma.masked_where(agentArray == 0, agentArray)
        color = random.choice(colors)
        colors.remove(color)
        agentColors.append(color)
        axis.imshow(agentArray, cmap=color, aspect='equal', origin='upper')
    timeSlider = Slider(axes, "Time", 0, maxTime, valstep=1.0, valinit=0.0)
    timeSlider.on_changed(updateMap)
    plt.show()

def createPath(parentInfo, start, end): #recreate the path based on recorded nodes
    path = []
    currentPos = end
    currentNode = parentInfo
    while currentNode[1] != None:
        path.append(currentNode[0])
        currentNode = currentNode[1]
    path.append(start)
    path.reverse()
    return path

def checkCollision(cPosition, nPosition, nodeInfo, closedNodes, constraints): #FIXXXXX
    fValue = nodeInfo[0]
    gValue = nodeInfo[1]
    for constraint in constraints:
        if onMap(nPosition) and nPosition not in mapWallSpaces and [nPosition[0], nPosition[1], gValue + 1] not in closedNodes:
            if constraint[2] is not None: #edge constriaint
                if cPosition == constraint[0] and gValue == constraint[1]:
                    if nPosition == constraint[2]:
                        return False
                elif cPosition == constraint[2] and gValue == constraint[1]:
                    if nPosition == constraint[0]:
                        return False
            elif constraint[2] is None: #vertex constraint
                if nPosition == constraint[0] and gValue + 1 == constraint[1]:
                    return False
    return True

def aStar(start, end, constraints): #A* algorithm
    global maxTime, starts, ends, AStarAllNodes
    if start is None and end is None:
        start, end = startAndEnd()
        while end == start or start in starts or end in ends:
            start, end = startAndEnd()
    starts.append(start)
    ends.append(end)
    # nodeInfo = [[Node() for _ in range(width)] for _ in range(height)]
    closedNodes = []
    openNodes = pq()
    openNodes2 = []
    openNodes.put((0, start))
    currAndParent = [start, None]
    hq.heappush(openNodes2, (heuristicValue(start, end), 0, currAndParent))
    closedNodes.append([start[0], start[1], 0])
    while not openNodes.empty():
        print(openNodes2)
        hq.heapify(openNodes2)
        fValue, gValue, parentInfo = hq.heappop(openNodes2)
        cPosition = parentInfo[0]
        info = [fValue, gValue]
        fValue = info[0]
        gValue = info[1]
        # _, cPosition = openNodes.get()
        if cPosition == end:
            path = createPath(parentInfo, start, end)
            return path
        closedNodes.append([cPosition[0], cPosition[1], gValue])
        neighborCells = [
            [cPosition[0] + 1, cPosition[1]],
            [cPosition[0] - 1, cPosition[1]],
            [cPosition[0], cPosition[1] + 1],
            [cPosition[0], cPosition[1] - 1],
            [cPosition[0], cPosition[1]]
        ]

        for nPosition in neighborCells:
            currAndParent = [nPosition, parentInfo]
            if nPosition == cPosition:
                newG = gValue + 1  # staying in place
                newH = heuristicValue(nPosition, end)
                newF = newG + newH
                if checkCollision(cPosition, nPosition, info, closedNodes, constraints):
                    hq.heappush(openNodes2, (newF, newG, currAndParent))
                    closedNodes.append([nPosition[0], nPosition[1], newG])
            if onMap(nPosition) and nPosition not in mapWallSpaces and [nPosition[0], nPosition[1], gValue + 1] not in closedNodes:
                if checkCollision(cPosition, nPosition, info, closedNodes, constraints):
                    newG = gValue + 1
                    newH = heuristicValue(nPosition, end)
                    newF = newG + newH
                    hq.heappush(openNodes2, (newF, newG, currAndParent))
                    closedNodes.append([nPosition[0], nPosition[1], newG])

    return []  # no path

class CBSNode: #node of CBS
    def __init__(self):
        self.cost = 0
        self.paths = []
        self.collisions = []
        self.constraints = []

    def __lt__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __ge__(self, other):
        return self.cost >= other.cost

    def __eq__(self, other):
        return self.cost == other.cost

    def __ne__(self, other):
        return self.cost != other.cost

def totalCost(agents):
    return sum(len(agent) for agent in agents)

def detectCollision(agent1, agent2): #takes in the paths, finds first new collision between the pair
    global allCollisions
    localCollision = []
    length = min(len(agent1), len(agent2))
    print('------------------------------------------------')
    for k in range(length):
        if agent1[k] in agent2:
            if agent1[k] == agent2[k]: #vertex collision
                print('collision 1 detected', k)
                print([agent1[k][0], agent1[k][1], k], allCollisions)
                x = agent1[k][0]
                y = agent1[k][1]
                allCollisions.append([x, y, k, 'vertex'])
                localCollision.append([[x, y], k, 'vertex', None, agent1, agent2])
                print("collision 1 at time", k)
                return localCollision
            else:
                try:
                    if agent1[k + 1] == agent2[k] and agent1[k] == agent2[k + 1]:  # edge collision
                        print('collision 2 detected', k)
                        print([agent1[k][0], agent1[k][1], k], allCollisions)
                        x = agent1[k][0]
                        y = agent1[k][1]
                        allCollisions.append([x, y, k, 'edge'])
                        localCollision.append([[x, y], k, 'edge', agent1[k + 1], agent1, agent2])
                        print("collision 2 at time", k)
                        return localCollision
                    else:
                        print('no collision')
                except:
                    print('no collision')
        else:
            print('no collision')


def detectAllCollisions(paths): #find all collisions in paths
    paths.sort(key=len, reverse=True)
    allCurrentCollisions = []
    if len(paths) > 1:
        for i in range(len(paths) - 1):
            for j in range(i + 1, len(paths)):
                collide = detectCollision(paths[i], paths[j])
                if collide:
                    allCurrentCollisions.extend(collide)
    return allCurrentCollisions

def standardSplitting(collision):
    if collision[2] == 'edge':
        return [collision[0], collision[1], collision[3]] #first edge collision location, time, second edge collision location
    elif collision[2] == 'vertex':
        return [collision[0], collision[1], None] #collision location, time

def CBS(paths):
    global allCollisions
    allConstraints = []
    nodesCBS = pq()
    initialNode = CBSNode()
    initialNode.paths = paths.copy()
    initialNode.cost = totalCost(paths)
    initialNode.collisions = detectAllCollisions(paths)
    nodesCBS.put((initialNode.cost, initialNode))
    while not nodesCBS.empty():
        currentNode = nodesCBS.get()[1]
        if not currentNode.collisions:
            return currentNode.paths
        currentCollision = currentNode.collisions[0]
        constraints = standardSplitting(currentCollision)
        allConstraints.append(constraints)
        for agent in currentCollision[-2:]:
            newPath = aStar(agent[0], agent[-1], allConstraints)
            print('constraints', allConstraints)
            print('path', newPath)
            if not newPath:  # no path
                continue
            pathIndex = currentNode.paths.index(agent)
            newNode = CBSNode()
            newPaths = currentNode.paths.copy()
            newPaths[pathIndex] = newPath
            newNode.paths = newPaths
            newNode.cost = totalCost(newPaths)
            newNode.collisions = detectAllCollisions(newPaths)
            newNode.constraints = allConstraints
            nodesCBS.put((newNode.cost, newNode))
            if not newNode.collisions:
                return newNode.paths
    return paths

# main
displayMap('specificTest.txt')
