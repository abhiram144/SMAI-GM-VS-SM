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


# In[7]:


import math
import warnings

def get_intra_cluster_distance(cluster_centroids,train_data_labels) :
    
    number_of_clusters = cluster_centroids.shape[0]
    
    dist_list = []
    for i in range(number_of_clusters) :
        dist_list.append(0)
    
    for i in range(trainVector.shape[0]) :
        
        cluster_number = train_data_labels[i]
        eucl_dist = cluster_centroids[cluster_number] - trainVector[i]
        eucl_dist = np.square(eucl_dist)
        eucl_dist = np.sum(eucl_dist)
        eucl_dist = math.sqrt(eucl_dist)
        
        dist_list[cluster_number] = dist_list[cluster_number] + eucl_dist
    
    return max(dist_list)


from scipy.spatial.distance import cdist, euclidean

def geometric_median_weinsfeild(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


# In[48]:


def get_inter_cluster_distance(cluster_centroids) :
    
    number_of_clusters = cluster_centroids.shape[0]
    intra_cluster_list = []
    
    for i in range(number_of_clusters) :
        for j in range(i+1,number_of_clusters) :
            eucl_dist = cluster_centroids[i] - cluster_centroids[j]
            eucl_dist = np.square(eucl_dist)
            eucl_dist = np.sum(eucl_dist)
            eucl_dist = math.sqrt(eucl_dist)
            
            intra_cluster_list.append(eucl_dist)
    
    return min(intra_cluster_list)
    


# In[51]:


def return_dunn_index(kmeans_object) :
    cluster_centroids = kmeans_object.cluster_centers_
    train_data_labels = kmeans_object.labels_
    
    #print(type(cluster_centroids))
    #print(type(train_data_labels))
    #print(cluster_centroids.shape)
    #print(train_data_labels.shape)
    #print(kmeans.inertia_)
    
    intra_cluster_dist = get_intra_cluster_distance(cluster_centroids,train_data_labels)
    inter_cluster_dist = get_inter_cluster_distance(cluster_centroids)
    
    print(inter_cluster_dist/intra_cluster_dist)


# In[52]:


return_dunn_index(kmeans)


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




