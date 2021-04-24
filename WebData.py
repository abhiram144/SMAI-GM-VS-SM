#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from time import process_time
from sklearn.cluster import KMeans
from sklearn.metrics import *
import datetime
pathWeb = "./Web/Web/data/"
import pickle


# In[2]:


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


# In[3]:


class GraphHelper:
    def __init__(self, trainData, mProtoTypes):
        self.trainData = trainData
        self.InitializeGraphToVector(trainData, mProtoTypes)
        self.mProtoTypes = mProtoTypes
        
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
            print(graphIndex)
            for nodeIndex,node in enumerate(nodes):
                print(nodeIndex, end = " ")
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
    
    def GraphToVector(self, graphSet):
        vectorMatrix = np.empty(shape= (len(graphSet), self.mProtoTypes))
        for row, graph in enumerate(graphSet):
            for col,prototypeIndex in enumerate(self.selectedProtoTypes):
                vectorMatrix[row][col] = self.GetDistanceBetweenGraphs(graph, self.trainData[prototypeIndex])
        return vectorMatrix
    
    def InitializeGraphToVector(self, graphSet, mprotoTypes):
        self.selectedProtoTypes = self.SelectSpanningPrototypes(graphSet, mprotoTypes)
        #return self.GraphToVector(graphSet)


# In[4]:


XtrainWeb, y_train = LoadData(pathWeb +"train.cxl", "fingerprints")
XvalidateWeb, y_validate = LoadData(pathWeb +"valid.cxl", "fingerprints")
XtestWeb, y_test = LoadData(pathWeb +"test.cxl", "fingerprints")


# In[5]:


t1_start = process_time() 
graph = GraphHelper(XtrainWeb, 10)
t1_stop = process_time()
print(t1_stop - t1_start)
now = datetime.datetime.now()
print("Completed Prototype Selection at ", str(now))


# In[ ]:


t1_start = process_time() 
trainVector = graph.GraphToVector(XtrainWeb)
t1_stop = process_time()
print(t1_stop - t1_start)
now = datetime.datetime.now()
print("Completed Train Conversion at ", str(now))


# In[ ]:


t1_start = process_time() 
testVector = graph.GraphToVector(XtestWeb)
t1_stop = process_time()
print(t1_stop - t1_start)
now = datetime.datetime.now()
print("Completed Test at ", str(now))


# In[ ]:


t1_start = process_time() 
validationVector = graph.GraphToVector(XvalidateWeb)
t1_stop = process_time()
print(t1_stop - t1_start)
now = datetime.datetime.now()
print("Completed Validation at ", str(now))


# ## Saving the computed Data

# In[ ]:


np.save("TrainVectorWeb", trainVector)
np.save("TestVectorWeb", testVector)
np.save("validateVectorWeb", validationVector)
with open('GraphHelperObjWeb', 'wb') as config_dictionary_file:
    pickle.dump(graph, config_dictionary_file)


# ## Loading from saved files 
# 

# In[ ]:


trainVector = np.load("TrainVectorWeb.npy")
testVector = np.load("TestVectorWeb.npy")
validationVector = np.load("validateVectorWeb.npy")
with open('GraphHelperObjWeb', 'rb') as config_dictionary_file:
    graph = pickle.load(config_dictionary_file)


# In[18]:


kmeans = KMeans(n_clusters=22, random_state=0).fit(trainVector)


# In[19]:


from sklearn.metrics.cluster import rand_score
score = rand_score(y_train,kmeans.predict(trainVector))
print('Accuracy of Train:{0:f}'.format(score))

score = rand_score(y_validate,kmeans.predict(validationVector))
print('Accuracy of Validation:{0:f}'.format(score))

score = rand_score(y_test,kmeans.predict(testVector))
print('Accuracy of Test:{0:f}'.format(score))


# In[20]:


from sklearn import svm
clf = svm.SVC()
clf.fit(trainVector, y_train)


# In[21]:


svmPredtrain = clf.predict(trainVector)
score = accuracy_score(y_train,svmPredtrain)
print('Accuracy:{0:f}'.format(score))


# In[22]:


svmPredvalidate = clf.predict(validationVector)
score = accuracy_score(y_validate,svmPredvalidate)
print('Accuracy Validate:{0:f}'.format(score))

svmPredtest = clf.predict(testVector)
score = accuracy_score(y_test,svmPredtest)
print('Accuracy Test:{0:f}'.format(score))


# In[23]:


print(testVector.shape)
print(validationVector.shape)
print(trainVector.shape)


# In[ ]:




