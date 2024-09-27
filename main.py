import time

import matplotlib.pyplot as plt
import numpy as np
import random
import heapq as hq
from matplotlib.animation import FuncAnimation

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
agents = 30
starts = []
ends = []
timeLimit = 5
test = [1, 4] #generate starting position for specific tests
display = plt.figure()


colors = ["spring", "summer", "autumn", "winter", "cool", "Wistia", "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "bwr",
          "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "tab10", "tab20", "tab20b", "tab20c", 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                      'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                      'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'] #agent colors

def onMap(pos):
    return 0 <= pos[0] < width and 0 <= pos[1] < height

def startAndEnd(): #generate starting and ending locations
    start = []
    end = []
    sPos = random.choice(mapPathSpace)
    ePos = random.choice(mapPathSpace)
    start.append(sPos[0])
    start.append(sPos[1])
    end.append(ePos[0])
    end.append(ePos[1])
    # sPos = random.choice(test)  # small area testing
    # if sPos == 1:
    #     start = [1, 1]
    #     end = [4, 1]
    # else:
    #     start = [4, 1]
    #     end = [1, 1]
    # sPos = random.choice(test)  # small area testing
    # if sPos == 1:
    #     start = [1, 1]
    #     end = [4, 1]
    # else:
    #     start = [4, 1]
    #     end = [1, 1]
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

def updateMap(frame):  # Update map for each frame
    global currentTime, axis, mapArray
    currentTime = frame
    axis.clear()
    agentArray = agentMap(None, None, True)
    agentArray = np.ma.masked_where(agentArray == 0, agentArray)
    axis.imshow(mapArray, cmap='grey', aspect='equal', origin='upper')
    axis.imshow(agentArray, cmap='binary', aspect='equal', origin='upper')
    for i in range(agents):
        agentPath = allPaths[i]
        color = agentColors[i]
        limit = min(len(agentPath), currentTime) + 1
        agentPos = agentPath[:limit][-1]  # Get current position
        agentArray = agentMap(agentPath, agentPos, False)
        agentArray = np.ma.masked_where(agentArray == 0, agentArray)
        axis.imshow(agentArray, cmap=color, aspect='equal', origin='upper')

def displayMap(mapFile):  # Starting command, generates agents, collisionless paths, and displays map and agents
    global allPaths, maxTime, axis, mapArray
    startT = time.time()
    pathLengths = []
    maxTime = 0
    mapArray = readMap(mapFile)
    fig, axis = plt.subplots()
    axis.imshow(mapArray, cmap='grey', aspect='equal', origin='upper')

    allAgents = []
    for i in range(agents):
        agentPath = aStar(None, None, [])
        allAgents.append(agentPath)

    finalAgentPaths = CBS(allAgents)
    allPaths = finalAgentPaths
    print("# of agents", len(allPaths))

    for path in allPaths:
        pathLengths.append(len(path) - 1)
        if len(path) > maxTime:
            maxTime = len(path) - 1

    for agentPath in allPaths:
        displayPos = agentPath[-1]
        agentArray = agentMap(agentPath, displayPos, False)
        agentArray = np.ma.masked_where(agentArray == 1, agentArray)
        color = random.choice(colors)
        colors.remove(color)
        agentColors.append(color)

    print('total time', round(time.time() - startT, 4))
    print('max length', max(pathLengths))
    print('min length', min(pathLengths))

    animation = FuncAnimation(fig, updateMap, frames=range(maxTime + 1), interval=20)
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
    closedNodes = []
    openNodes = []
    currAndParent = [start, None]
    hq.heappush(openNodes, (heuristicValue(start, end), 0, currAndParent))
    closedNodes.append([start[0], start[1], 0])
    while len(openNodes) != 0:
        hq.heapify(openNodes)
        fValue, gValue, parentInfo = hq.heappop(openNodes)
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
                    hq.heappush(openNodes, (newF, newG, currAndParent))
                    closedNodes.append([nPosition[0], nPosition[1], newG])
            if onMap(nPosition) and nPosition not in mapWallSpaces and [nPosition[0], nPosition[1], gValue + 1] not in closedNodes:
                if checkCollision(cPosition, nPosition, info, closedNodes, constraints):
                    newG = gValue + 1
                    newH = heuristicValue(nPosition, end)
                    newF = newG + newH
                    hq.heappush(openNodes, (newF, newG, currAndParent))
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
    localCollision = []
    minLength = min(len(agent1), len(agent2))
    maxLength = max(len(agent1), len(agent2))
    # if minLength == len(agent1):
    #     while len(agent1) != len(agent2):
    #         agent1.append(agent1[-1])
    # elif minLength == len(agent2):
    #     while len(agent2) != len(agent1):
    #         agent2.append(agent2[-1])
    print('------------------------------------------------')
    for k in range(minLength):
        if agent1[k] in agent2:
            if agent1[k] == agent2[k]: #vertex collision
                print('collision 1 detected', k)
                x = agent1[k][0]
                y = agent1[k][1]
                localCollision.append([[x, y], k, 'vertex', None, agent1, agent2])
                print("collision 1 at time", k)
                return localCollision
            else:
                try:
                    if agent1[k + 1] == agent2[k] and agent1[k] == agent2[k + 1]:  # edge collision
                        print('collision 2 detected', k)
                        x = agent1[k][0]
                        y = agent1[k][1]
                        localCollision.append([[x, y], k, 'edge', agent1[k + 1], agent1, agent2])
                        print("collision 2 at time", k)
                        return localCollision
                    else:
                        print('no collision')
                except: #incase of index error
                    print('no collision', k)
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
    nodeExploredCount = 0
    allConstraints = []
    nodesCBS = []
    initialNode = CBSNode()
    initialNode.paths = paths.copy()
    initialNode.cost = totalCost(paths)
    initialNode.collisions = detectAllCollisions(paths)
    hq.heappush(nodesCBS, (initialNode.cost, initialNode))
    while len(nodesCBS) != 0:
        _, currentNode = hq.heappop(nodesCBS)
        nodeExploredCount += 1
        if not currentNode.collisions:
            print('num of constraints', len(allConstraints))
            print('number of nodes', nodeExploredCount)
            return currentNode.paths
        currentCollision = currentNode.collisions[0]
        constraint = standardSplitting(currentCollision)
        allConstraints.append(constraint)
        for agent in currentCollision[-2:]:
            newPath = aStar(agent[0], agent[-1], allConstraints)
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
            hq.heappush(nodesCBS, (newNode.cost, newNode))
            if newNode.collisions == []:
                nodeExploredCount += 1
                print('num of constraints', len(allConstraints))
                print('number of nodes', nodeExploredCount)
                return newNode.paths

    return paths

# main
displayMap('warehouse.txt')
