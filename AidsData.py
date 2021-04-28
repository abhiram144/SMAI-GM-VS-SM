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
pathAids = "./AIDS/AIDS/data/"
path1Grec = "./GREC/GREC/data/"
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


XtrainAids, y_train = LoadData(pathAids +"train.cxl", "fingerprints")
XvalidateAids, y_validate = LoadData(pathAids +"valid.cxl", "fingerprints")
XtestAids, y_test = LoadData(pathAids +"test.cxl", "fingerprints")


# In[5]:


# t1_start = process_time() 
# graph = GraphHelper(XtrainAids, 10)
# t1_stop = process_time()
# print(t1_stop - t1_start)
# now = datetime.datetime.now()
# print("Completed Prototype Selection at ", str(now))


# # In[6]:


# t1_start = process_time() 
# trainVector = graph.GraphToVector(XtrainAids)
# t1_stop = process_time()
# print(t1_stop - t1_start)
# now = datetime.datetime.now()
# print("Completed Train Conversion at ", str(now))


# # In[7]:


# t1_start = process_time() 
# testVector = graph.GraphToVector(XtestAids)
# t1_stop = process_time()
# print(t1_stop - t1_start)
# now = datetime.datetime.now()
# print("Completed Test at ", str(now))


# # In[8]:


# t1_start = process_time() 
# validationVector = graph.GraphToVector(XvalidateAids)
# t1_stop = process_time()
# print(t1_stop - t1_start)
# now = datetime.datetime.now()
# print("Completed Validation at ", str(now))


# # ## Saving the computed Data

# # In[9]:


# np.save("TrainVectorAids", trainVector)
# np.save("TestVectorAids", testVector)
# np.save("validateVectorAids", validationVector)
# with open('GraphHelperObjAids', 'wb') as config_dictionary_file:
#     pickle.dump(graph, config_dictionary_file)


# ## Loading from saved files 
# 

# In[5]:


trainVector = np.load("TrainVectorAids.npy")
testVector = np.load("TestVectorAids.npy")
validationVector = np.load("validateVectorAids.npy")
with open('GraphHelperObjAids', 'rb') as config_dictionary_file:
    graph = pickle.load(config_dictionary_file)


# In[8]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(trainVector)

from sklearn.metrics.pairwise import euclidean_distances

def delta(ck, cl):
    values = np.ones([len(ck), len(cl)])*10000
    
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])
            
    return np.min(values)

def dunn(k_list):
    """ Dunn index [CVI]
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    deltas = np.ones([len(k_list), len(k_list)])*1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])
        
        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas)/np.max(big_deltas)
    return di

def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])
    
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])
            
    return np.max(values)
# In[7]:
import math
import pandas as pd
from sklearn.utils.multiclass import unique_labels
def InterClusterDistance(cluster_labels, data) :
    uniqueLabels = unique_labels(cluster_labels)
    globalMin = sys.maxsize
    for i, label in enumerate(uniqueLabels):
        for j in range(i + 1, len(uniqueLabels)):
            clusterPointIndices = np.where(cluster_labels == label)
            otherClusterIndices = np.where(cluster_labels == uniqueLabels[j])
            clusterPoints = data[clusterPointIndices]
            otherClusterPoints = data[otherClusterIndices]
            pairWiseDifference = np.square(clusterPoints[:, np.newaxis] - otherClusterPoints)
            pairWiseDifferenceReshaped = pairWiseDifference.reshape(-1, clusterPoints.shape[1])
            pairWiseEucledian = np.sqrt(pairWiseDifferenceReshaped.sum(axis = 1))
            clusterMin = pairWiseEucledian.min()
            if(clusterMin < globalMin):
                globalMin = clusterMin
    return globalMin

def IntraClusterDistance(cluster_labels, data):
    globalMax = 0
    uniqueLabels = unique_labels(cluster_labels)
    pred = pd.DataFrame(cluster_labels)
    pred.columns = ['Type']
    df = pd.DataFrame(data)
    # we merge this dataframe with df
    prediction = pd.concat([df, pred], axis = 1)
    for label in uniqueLabels:
        clusterPointIndices = np.where(cluster_labels == label)
        clusterPoints = data[clusterPointIndices]
        pairWiseDifference = np.square(clusterPoints[:, np.newaxis] - clusterPoints)
        pairWiseDifferenceReshaped = pairWiseDifference.reshape(-1, clusterPoints.shape[1])
        pairWiseEucledian = np.sqrt(pairWiseDifferenceReshaped.sum(axis = 1))
        clusterMax = pairWiseEucledian.max()
        if(clusterMax > globalMax):
            globalMax = clusterMax
    return globalMax



def DunnScoreCalculation(kmeans_object, trainData, givenLabels = None):
    cluster_centroids = kmeans_object.cluster_centers_
    if(givenLabels is None):
        train_data_labels = kmeans_object.labels_
    else:
        train_data_labels = givenLabels
    # interClusterDistance = InterClusterDistance(train_data_labels, trainData)
    # intraClusterDistance = IntraClusterDistance(train_data_labels, trainData)
    # minDunnScore = np.min(interClusterDistance / intraClusterDistance)
    # return minDunnScore

    df = pd.DataFrame(trainData)
  
    # K-Means
    from sklearn import cluster
    y_pred = train_data_labels
    
    # We store the K-means results in a dataframe
    pred = pd.DataFrame(y_pred)
    pred.columns = ['Type']
    
    # we merge this dataframe with df
    prediction = pd.concat([df, pred], axis = 1)
    
    # We store the clusters
    clus0 = prediction.loc[prediction.Type == 0]
    clus1 = prediction.loc[prediction.Type == 1]
    cluster_list = [clus0.values, clus1.values]
    
    return dunn(cluster_list)


# In[52]:



# In[13]:


from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import homogeneity_score

pred = kmeans.predict(trainVector)
score = rand_score(y_train,pred)
print('Dunn Index Accuracy of Train:{0:f}'.format(score))
print("Homogenity Score of Train %.6f" % homogeneity_score(y_train, pred))

print()
print()
pred = kmeans.predict(validationVector)
score = rand_score(y_validate,pred)
print('Dunn Index Accuracy of Validation:{0:f}'.format(score))
print("Homogenity Score of Validation %.6f" % homogeneity_score(y_validate, pred))



print()
print()
pred = kmeans.predict(testVector)
score = rand_score(y_test,pred)
print('Dunn Index Accuracy of Test:{0:f}'.format(score))
print("Homogenity Score of Test %.6f" % homogeneity_score(y_test, pred))
from sklearn.preprocessing import LabelEncoder  
le = LabelEncoder()
y_train_labels = le.fit_transform(y_test)
DunnScoreCalculation(kmeans, testVector, y_train_labels)




# In[13]:


from sklearn import svm
clf = svm.SVC()
clf.fit(trainVector, y_train)


# In[14]:


svmPredtrain = clf.predict(trainVector)
score = accuracy_score(y_train,svmPredtrain)
print('Accuracy:{0:f}'.format(score))


# In[15]:


svmPredvalidate = clf.predict(validationVector)
score = accuracy_score(y_validate,svmPredvalidate)
print('Accuracy Validate:{0:f}'.format(score))

svmPredtest = clf.predict(testVector)
score = accuracy_score(y_test,svmPredtest)
print('Accuracy Test:{0:f}'.format(score))


# In[16]:


print(testVector.shape)
print(validationVector.shape)
print(trainVector.shape)


# In[8]:


geometric_median_weinsfeild(trainVector)


# In[ ]:




