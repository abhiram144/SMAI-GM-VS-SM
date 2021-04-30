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
pathGrec = "./GREC/GREC/data/"
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


XtrainGREC, y_train = LoadData(pathGrec +"train.cxl", "grec")
XvalidateGREC, y_validate = LoadData(pathGrec +"valid.cxl", "grec")
XtestGREC, y_test = LoadData(pathGrec +"test.cxl", "grec")


# In[5]:


t1_start = process_time() 
graph = GraphHelper(XtrainGREC, 10)
t1_stop = process_time()
print(t1_stop - t1_start)
now = datetime.datetime.now()
print("Completed Prototype Selection at ", str(now))


# In[6]:


t1_start = process_time() 
trainVector = graph.GraphToVector(XtrainGREC)
t1_stop = process_time()
print(t1_stop - t1_start)
now = datetime.datetime.now()
print("Completed Train Conversion at ", str(now))


# In[7]:


t1_start = process_time() 
testVector = graph.GraphToVector(XtestGREC)
t1_stop = process_time()
print(t1_stop - t1_start)
now = datetime.datetime.now()
print("Completed Test at ", str(now))


# In[8]:


t1_start = process_time() 
validationVector = graph.GraphToVector(XvalidateGREC)
t1_stop = process_time()
print(t1_stop - t1_start)
now = datetime.datetime.now()
print("Completed Validation at ", str(now))


# ## Saving the computed Data

# In[9]:


np.save("TrainVectorGREC", trainVector)
np.save("TestVectorGREC", testVector)
np.save("validateVectorGREC", validationVector)
with open('GraphHelperObjGREC', 'wb') as config_dictionary_file:
    pickle.dump(graph, config_dictionary_file)


# ## Loading from saved files 
# 

# In[41]:


trainVector = np.load("TrainVectorGREC.npy")
testVector = np.load("TestVectorGREC.npy")
validationVector = np.load("validateVectorGREC.npy")
with open('GraphHelperObjGREC', 'rb') as config_dictionary_file:
    graph = pickle.load(config_dictionary_file)


# In[62]:


from sklearn.metrics.pairwise import euclidean_distances

def delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]

    return np.min(values)
    
def big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    #values = values[np.nonzero(values)]
            
    return np.max(values)

def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    
    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = euclidean_distances(points)
    ks = np.sort(np.unique(labels))
    
    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])
    
    l_range = list(range(0, len(ks)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        
        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas)/np.max(big_deltas)
    return di


# In[63]:


kmeans = KMeans(n_clusters=22, random_state=0).fit(trainVector)


# In[64]:


from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import homogeneity_score

pred = kmeans.predict(trainVector)
score = rand_score(y_train,pred)
print('Rand Index Accuracy of Train:{0:f}'.format(score))
print("Homogenity Score of Train %.6f" % homogeneity_score(y_train, pred))
print('Dunn Index of the Cluster is :{0:f}'.format(dunn_fast(trainVector, pred)))
le = LabelEncoder()
y_train_labels = le.fit_transform(y_train)
print('Dunn Index of the Ground Truth is :{0:f}'.format(dunn_fast(trainVector, y_train_labels)))



print()
print()
pred = kmeans.predict(validationVector)
score = rand_score(y_validate,pred)
print('Rand Index Accuracy of Validation:{0:f}'.format(score))
print("Homogenity Score of Validation %.6f" % homogeneity_score(y_validate, pred))
print('Dunn Index of the Cluster is :{0:f}'.format(dunn_fast(validationVector, pred)))
le = LabelEncoder()
y_validate_labels = le.fit_transform(y_validate)
print('Dunn Index of the Ground Truth is :{0:f}'.format(dunn_fast(validationVector, y_validate_labels)))

print()
print()
pred = kmeans.predict(testVector)
score = rand_score(y_test,pred)
print('Rand Index Accuracy of Test:{0:f}'.format(score))
print("Homogenity Score of Test %.6f" % homogeneity_score(y_test, pred))
print('Dunn Index of the Cluster is :{0:f}'.format(dunn_fast(testVector, pred)))

le = LabelEncoder()
y_train_labels = le.fit_transform(y_test)
print('Dunn Index of the Ground Truth is :{0:f}'.format(dunn_fast(testVector, y_train_labels)))


# # Generalized Median

# In[67]:


class GraphHelperGM:
    def __init__(self, trainData):
        self.trainData = trainData
        
    def GraphToVector(self,graphSet):
        ged_env = GEDEnv() # initailize GED environment.
        ged_env.set_edit_cost('CONSTANT', # GED cost type.
                            edit_cost_constants=[3, 3, 1, 3, 3, 1] # edit costs.
                            )  
        for graph in graphSet:
            ged_env.add_nx_graph(graph, '') # add graph1
        for graph in self.trainData:
            ged_env.add_nx_graph(graph, '') # add graph1
        listID = ged_env.get_all_graph_ids() # get list IDs of graphs
        ged_env.init(init_type='LAZY_WITHOUT_SHUFFLED_COPIES') # initialize GED environment.
        options = {'initialization_method': 'RANDOM', # or 'NODE', etc.
                'threads': 1 # parallel threads.
                }
        ged_env.set_method('BIPARTITE', # GED method.
                        options # options for GED method.
                        )
        ged_env.init_method() # initialize GED method.
        
        
        vectorMatrix = np.empty(shape= (len(graphSet), len(self.trainData)))
        for row, graph in enumerate(graphSet):
            for col  in range(len(self.trainData)):
                ged_env.run_method(listID[row], listID[len(graphSet) + col]) # run.
                dis = ged_env.get_upper_bound(listID[row], listID[len(graphSet) + col])
                vectorMatrix[row][col] = dis
        return vectorMatrix


# In[68]:


graphGM = GraphHelperGM(XtrainGREC)


# In[69]:


t1_start = process_time() 
trainVectorGM = graphGM.GraphToVector(XtrainGREC)
t1_stop = process_time()
print(t1_stop - t1_start)
now = datetime.datetime.now()
print("Completed Test at ", str(now))


# In[70]:


t1_start = process_time() 
testVectorGM = graphGM.GraphToVector(XtestGREC)
t1_stop = process_time()
print(t1_stop - t1_start)
now = datetime.datetime.now()
print("Completed Test at ", str(now))


# In[ ]:


t1_start = process_time() 
validationVectorGM = graphGM.GraphToVector(XvalidateGREC)
t1_stop = process_time()
print(t1_stop - t1_start)
now = datetime.datetime.now()
print("Completed Test at ", str(now))


# ## Saving Data Grec GM

# In[50]:


np.save("TrainVectorGRECGM", trainVectorGM)
np.save("TestVectorGRECGM", testVectorGM)
np.save("validateVectorGRECGM", validationVectorGM)
with open('GraphHelperObjGRECGM', 'wb') as config_dictionary_file:
    pickle.dump(graphGM, config_dictionary_file)


# ## Loading Data Grec GM

# In[51]:


trainVectorGM = np.load("TrainVectorGRECGM.npy")
testVectorGM = np.load("TestVectorGRECGM.npy")
validationVectorGM = np.load("validateVectorGRECGM.npy")
with open('GraphHelperObjGRECGM', 'rb') as config_dictionary_file:
    graphGM = pickle.load(config_dictionary_file)


# In[66]:


from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.preprocessing import LabelEncoder  


kmeansGM = KMeans(n_clusters=22, random_state=0).fit(trainVectorGM)
pred = kmeansGM.predict(trainVectorGM)
score = rand_score(y_train,pred)
print('Rand Index Accuracy of Train:{0:f}'.format(score))
print("Homogenity Score of Train %.6f" % homogeneity_score(y_train, pred))
print('Dunn Index of the Cluster is :{0:f}'.format(dunn_fast(trainVectorGM, pred)))
le = LabelEncoder()
y_train_labels = le.fit_transform(y_train)
print('Dunn Index of the Ground Truth is :{0:f}'.format(dunn_fast(trainVectorGM, y_train_labels)))


print()
print()
pred = kmeansGM.predict(validationVectorGM)
score = rand_score(y_validate,pred)
print('Rand Index Accuracy of Validation:{0:f}'.format(score))
print("Homogenity Score of Validation %.6f" % homogeneity_score(y_validate, pred))
print('Dunn Index of the Cluster is :{0:f}'.format(dunn_fast(validationVectorGM, pred)))
le = LabelEncoder()
y_validate_labels = le.fit_transform(y_validate)
print('Dunn Index of the Ground Truth is :{0:f}'.format(dunn_fast(validationVectorGM, y_validate_labels)))

print()
print()
pred = kmeansGM.predict(testVectorGM)
score = rand_score(y_test,pred)
print('Rand Index Accuracy of Test:{0:f}'.format(score))
print("Homogenity Score of Test %.6f" % homogeneity_score(y_test, pred))
print('Dunn Index of the Cluster is :{0:f}'.format(dunn_fast(testVectorGM, pred)))
le = LabelEncoder()
y_train_labels = le.fit_transform(y_test)
print('Dunn Index of the Ground Truth is :{0:f}'.format(dunn_fast(testVectorGM, y_train_labels)))


# In[ ]:




