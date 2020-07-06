import vsm
import numpy as np
from random import randint


class kmeans:
    
    '''
    Random Centroid Selection
    Returns randomly selected centroids
    '''
    def random_seed(self):
        #index = [0,168,311,630,713]
        centroids = []
        for i in range(0,self.K):
            index = randint(0, len(self.vectors))
            centroids.append(self.vectors[index])
            #print(index)
        return centroids
    
    '''
    Euclidean Distance Calculation Between Two document vectors
    Inputs two document vectors
    Returns cosine similarity between them
    '''
    def euclidean_distance(self, a, b):
        dist = np.linalg.norm(a-b)
        return dist
        
    '''
    Cosine Similarity Calculation Between Two document vectors
    Inputs two document vectors
    Returns cosine similarity between them
    '''
    def cosine_sim(self, a, b):
        cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        return cos_sim

    '''
    Recalculation of Centroids after each clustering iteration
    Inputs Label Dictionary
    Returns new centroids
    '''
    def calulate_centroids(self, labels):
        centroid = []
        for i in range(len(labels)):
            x = labels[i]
            c = np.zeros(self.vectors.shape[1], dtype = float) 
            for j in range(0,len(x)):
                c += np.array(self.vectors[x[j]])
            centroid.append(c/len(x))
    
        return centroid
        
    
    '''
    Performs clustering 
    Input Documents, K clusters, maximum iterations, and minimum document frequency
    Return Clusters assigned to each documents
    '''
    def clustering(self, X, k, max_iterations = 10, min_df = 0):
        self.tfidf = vsm.TF_IDF()
        self.K = k 
        self.N = X.shape[0]
        self.vectors = self.tfidf.tfidf(X, min_df)
        self.features = self.tfidf.features

        centroids = self.random_seed()
        
        labels = {}
        
        prev = [-1] * self.N
        y_pred = [0] * self.N
        for iteration in range(0, max_iterations):
            
            if (y_pred == prev):
                #print("Iteration #" + str(iteration))
                break
            
            prev = y_pred
            
            for i in range(0,self.K):
                labels[i] = []
                            
            for i in range(0,self.N):
                distance = []
                for j in range(0,self.K):
                    distance.append(self.cosine_sim(np.array(self.vectors[i]),np.array(centroids[j])))
                
                index = distance.index(max(distance))
                labels[index].append(i)
            
            centroids = self.calulate_centroids(labels)
            
            #print("Iteration #" + str(iteration))
            
            y_pred = [0] * self.N
            for i in labels:
                for j in labels[i]:
                    y_pred[j] = i
        
        return y_pred, labels