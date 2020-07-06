from nltk.tokenize import word_tokenize 
import re
import math
from nltk.stem import PorterStemmer
from collections import Counter
import numpy as np
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
#import spacy
#spacy_nlp = spacy.load('en_core_web_sm')

#Stemmer and Lemmatizer initialization
lemmatizer = WordNetLemmatizer() 
ps = PorterStemmer() 

#Stopword Initialization
stop_words = stopwords.words('english')


class TF_IDF:
    
    DF = {}
    N = 0
    total_vocab = []
    
    '''
    Preprocessing Function takes file string or query string and 
    remove stopwords, special chracters, and lemmatize it
    and return tokens
    '''
    def preprocess(self,document):
        document = document.lower()
        document = re.sub(r'[^(?u)\b\w\w+\b]',' ', document)
        document = re.sub("[^a-z0-9]+"," ",document)
        tokens = word_tokenize(document)
        #tokens = spacy_nlp(document)
        #tokens = [token.lemma_ for token in doci]
        token = []
        for word in tokens:
                if word not in stop_words:
                    token.append(ps.stem(word))
        return token
    
    '''
    Return Document Frequency of each word in dictionary
    '''
    def doc_freq(self,word):
            c = 0
            try:
                c = self.DF[word]
            except:
                pass
            return c
        
    '''
    Takes input documents and minimum document frequency count
    Returns Document Vectors
    '''
    def tfidf(self, X, min_df = 0):
        
        processed_text = []
        self.N = X.shape[0]
        for i in range(self.N):
            news = X.iloc[i]
            news = self.preprocess(news)
    
            processed_text.append(news) 
        
        for i in range(self.N):
            tokens = processed_text[i]
            for w in tokens:
                try:
                    self.DF[w].add(i)
                except:
                    self.DF[w] = {i}
        
        d = self.DF
        for i in d:
            if(len(d[i]) < min_df):
                self.DF[i] = 0
            else:
                self.DF[i] = len(d[i])
        
        self.Dictionary = {}
        for i in self.DF:
            if self.DF[i] > 0:
                self.Dictionary[i] = self.DF[i]
        doc = 0
        
        self.features = self.Dictionary.keys()
        
        self.tf_idf = {}
        doc = 0
        for i in range(self.N):
            
            tokens = processed_text[i]
            
            counter = Counter(tokens)
            words_count = len(tokens)
            
            for token in np.unique(tokens):
                try:
                    tf = counter[token]/words_count
                    df = self.doc_freq(token)
                    idf = math.log10((self.N)/(df))
                    
                    self.tf_idf[doc, token] = tf*idf
                except:
                    pass
        
            doc += 1
           
        self.total_vocab = [x for x in self.Dictionary]
        total_vocab_size = len(self.Dictionary)
        self.D = np.zeros((self.N, total_vocab_size))
        for i in self.tf_idf:
            try:
                ind = self.total_vocab.index(i[1])
                self.D[i[0]][ind] = self.tf_idf[i]
            except:
                pass
            
        return self.D
    
    '''
    Takes input test document
    Returns Test Document Vector
    '''
    def query_vector(self,query):
            tokens = self.preprocess(query)
            Test_vector = np.zeros((len(self.total_vocab)))
            
            counter = Counter(tokens)
            words_count = len(tokens)
        
            
            for token in np.unique(tokens):
                try:
                    tf = counter[token]/words_count
                    df = self.doc_freq(token)
                    idf = math.log10((self.N)/(df))
                    ind = self.total_vocab.index(token)
                    Test_vector[ind] = tf*idf
                except:
                    pass
            return Test_vector
    

