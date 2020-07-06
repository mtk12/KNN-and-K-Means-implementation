import vsm
import numpy as np

class KNN:
    
    prediction = []
    features = []
    
    '''
    Cosine Similarity Calculation Between Two document vectors
    Inputs two document vectors
    Returns cosine similarity between them
    '''
    def cosine_sim(self, a, b):
        cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        return cos_sim

    '''
    Initialization of document vectors
    Inputs Training documents and minimum document frequency
    '''
    def train(self, X_train, min_df = 0):
        self.tfidf = vsm.TF_IDF()
        self.vectors = self.tfidf.tfidf(X_train, min_df)
        self.features = list(self.tfidf.features)
        self.N = X_train.shape[0]
    
    '''
    Prediction of labels of test documents 
    Inputs test documents, train labels and K nearest neighbours
    Return prediction of labels of test documents
    '''
    def predict(self, X_test, y_train, k = 3):
        self.prediction = []
        n = X_test.shape[0]
        
        for i in range(n):
            t = X_test.iloc[i]
            test = self.tfidf.query_vector(t)
            res=[]
            vec1 = test
            
            for x in range(self.N):
                vec2 = self.vectors[x]
                res.append(self.cosine_sim(vec1,vec2))
            
            res = np.argsort(res)
            labels = []
            for i in range(0,k):
                labels.append(y_train.iloc[res[-i]])
                
            self.prediction.append(max(set(labels), key = labels.count))
        
        return self.prediction