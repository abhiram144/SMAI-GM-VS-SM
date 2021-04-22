import enum
import sys
from graphkitlearn.graphkitlearn.gklearn.utils import graphfiles
import networkx
import matplotlib.pyplot as plt
import numpy as np
from gklearn.utils import *
import os
import random
from gklearn.ged.env import GEDEnv
import numpy as np

pathAids = "./AIDS/AIDS/data/"
path1Grec = "./GREC/GREC/data/"


def LoadData(filename, childrentagName):
    import xml.etree.ElementTree as ET
    dirname_dataset = os.path.dirname(filename)
    tree = ET.parse(filename)
    root = tree.getroot()
    data = []
    y = []
    children = list([elem for elem in root.find(childrentagName).iter() if elem is not root.find(childrentagName)])
    for graph in children:
        mol_filename = graph.attrib['file']
        mol_class = graph.attrib['class']
        data.append(graphfiles.loadGXL(dirname_dataset + '/' + mol_filename))
        y.append(mol_class)
    return data, y

class GraphHelper:
    def GetDistanceBetweenGraphs(self,graph1, graph2):
        ged_env = GEDEnv() # initailize GED environment.
        ged_env.set_edit_cost('CONSTANT', # GED cost type.
                            edit_cost_constants=[3, 3, 1, 3, 3, 1] # edit costs.
                            )  
        ged_env.add_nx_graph(graph1, '') # add graph1
        ged_env.add_nx_graph(graph2, '') # add graph2
        listID = ged_env.get_all_graph_ids() # get list IDs of graphs
        ged_env.init(init_type='LAZY_WITHOUT_SHUFFLED_COPIES') # initialize GED environment.
        options = {'initialization_method': 'RANDOM', # or 'NODE', etc.
                'threads': 1 # parallel threads.
                }
        ged_env.set_method('BIPARTITE', # GED method.
                        options # options for GED method.
                        )
        ged_env.init_method() # initialize GED method.

        ged_env.run_method(listID[0], listID[1]) # run.
        dis = ged_env.get_upper_bound(listID[0], listID[1])
        return dis

    def MaxEditDistance(self, graphSets, nodes, addedIndices):
        distanceVector = np.empty(shape=(len(graphSets ), len(nodes)))
        for graphIndex,graph in enumerate(graphSets):
            for nodeIndex,node in enumerate(nodes):
                dis = self.GetDistanceBetweenGraphs(graph, node)
                distanceVector[graphIndex][nodeIndex] = dis
        maxValue = -1
        maxIndex = -1
        for graphIndex in range(len(graphSets)):
            if(graphIndex not in addedIndices):
                maxDistanceIndex = np.argmax(distanceVector[graphIndex])
                if(distanceVector[graphIndex][maxDistanceIndex] > maxValue):
                    maxValue = distanceVector[graphIndex][maxDistanceIndex]
                    maxIndex = graphIndex

        return maxIndex

    def SelectSpanningPrototypes(self, graphData, mprototypes):
        choiceIndex = random.randrange(len(graphData))
        graphSelected = [graphData[choiceIndex]]
        graphSelectedIndex = [choiceIndex]

        for selectors in range(mprototypes - 1):
            maxEditDistanceIndex = self.MaxEditDistance(graphData, graphSelected, graphSelectedIndex)
            graphSelectedIndex.append(maxEditDistanceIndex)
            graphSelected.append(graphData[maxEditDistanceIndex])
        return graphSelectedIndex

    def GraphToVector(self, graphSet, mprotoTypes):
        self.selectedProtoTypes = self.SelectSpanningPrototypes(graphSet, mprotoTypes)
        vectorMatrix = np.empty(shape= (len(graphSet), mprotoTypes))
        for row, graph in enumerate(graphSet):
            for col,prototypeIndex in enumerate(self.selectedProtoTypes):
                vectorMatrix[row][col] = self.GetDistanceBetweenGraphs(graph, graphSet[prototypeIndex])
        return vectorMatrix






XtrainAids, y_train = LoadData(pathAids +"train.cxl", "fingerprints")
XvalidateAids, y_validate = LoadData(pathAids +"valid.cxl", "fingerprints")
XtestAids, y_test = LoadData(pathAids +"test.cxl", "fingerprints")
graph = GraphHelper()
trainVector = graph.GraphToVector(XtrainAids, 10)

