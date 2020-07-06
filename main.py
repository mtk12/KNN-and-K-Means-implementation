import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import KNN
import k_means
import time
from sklearn import metrics
import numpy as np

'''
Puttiing all data together in excel file
'''
'''
text = []
label = []
path = "bbcsport"
directory_path = os.path.join(os.getcwd(),path)
path = os.path.join(directory_path,"athletics")
for filename in glob.glob(os.path.join(path, '*.txt')):
    with open(filename,'r') as f:
        text.append(f.read())
        label.append("athletics")

path = os.path.join(directory_path,"cricket")
for filename in glob.glob(os.path.join(path, '*.txt')):
    with open(filename,'r') as f:
        text.append(f.read())
        label.append("cricket")

path = os.path.join(directory_path,"football")
for filename in glob.glob(os.path.join(path, '*.txt')):
    with open(filename,'r') as f:
        text.append(f.read())
        label.append("football")
        
path = os.path.join(directory_path,"rugby")
for filename in glob.glob(os.path.join(path, '*.txt')):
    with open(filename,'r') as f:
        text.append(f.read())
        label.append("rugby")
    
path = os.path.join(directory_path,"tennis")
for filename in glob.glob(os.path.join(path, '*.txt')):
    with open(filename,'r') as f:
        text.append(f.read())
        label.append("tennis")
        
df = pd.DataFrame(list(zip(text,label)), columns = ['text','label'])
df.to_excel("bbcsports.xlsx")
'''

'''
Computes Purity for K-means Clustering
'''
def purity_score(y, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

if __name__ == "__main__":
    '''
    Importing Data and seperating Documents and labels
    '''
    df = pd.read_excel("bbcsports.xlsx",index_col=0)
    X = df['text']
    y = df['label']
    
    '''
    Train and Test Split with equal train/test split from each class
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y)
    
    '''
    K-Nearest Neighbour Algorithm
    '''
    
    print("K-Nearest Neighbour")
    k = input("Enter K: ")
    print("=================================================================")
    k = int(k)
    start = time.time()
    knn = KNN.KNN()
    # Min Document Frequency is used as Feature Selcection Parameter
    knn.train(X_train, min_df = 0)
    y_pred = knn.predict(X_test, y_train, k)
    
    # Vectors and Features of X_train in KNN 
    '''
    knn_features = knn.features
    knn_vectors = knn.vectors
    '''
    
    score = 0
    for i in range(0,len(y_pred)):
        if y_pred[i] == y_test.iloc[i]:
            score += 1
    
    accuracy = score/len(y_pred)
    print("Accuracy: " + str(accuracy))
    end = time.time()
    tot = end - start
    print("Total Execution Time for KNN: " + str(int(tot)) + " seconds")
    print("=================================================================")
    
    
    '''
    K-Means Algorithm
    '''
    
    print("\n\nK-means Clustering")
    k = input("Enter K: ")
    max_iterations = input("Enter Maximum iterations: ")
    min_df = input("Enter Minimum document Frequency: ")
    print("=================================================================")
    k = int(k)
    max_iterations = int(max_iterations)
    min_df = int(min_df)
    
    start = time.time()
    Kmeans = k_means.kmeans()
    
    # Min Document Frequency is used as Feature Selcection Parameter
    y_pred, labels = Kmeans.clustering(X, k, max_iterations, min_df)
    
    # Vectors and Features of X in K-means Clustering 
    '''
    k_means_vector = Kmeans.vectors
    k_means_features = Kmeans.features
    '''
    
    contingency_matrix = metrics.cluster.contingency_matrix(y, y_pred)
    score = purity_score(y,y_pred)
    print("Purity: " + str(score))
    end = time.time()
    tot = end - start
    print("Total Execution Time for K-Means: " + str(int(tot)) + " seconds")
    print("=================================================================")



