from util import *
import numpy as np
from collections import Counter


class LSA():

    def __init__(self):

        self.rank_approximation = 150

    def __buildDF(self, docs, docIDs):

        DF = {}
        for doc, doc_id in zip(docs, docIDs):
            for sent in doc:
                for word in sent:
                    if word in DF:
                        DF[word].add(doc_id)
                    else:
                        DF[word] = {doc_id}

        for term in DF:
            DF[term] = len(DF[term])

        vocab = [term for term in DF]

        self.vocab = vocab

        self.DF = DF

    def __calucateTF_IDF(self, docs, docIDs):

        vocab_size = len(self.vocab)
        no_of_docs = len(docIDs)

        tf_idf = {}
        for doc, doc_id in zip(docs, docIDs):
            words = []
            for sent in doc:
                for word in sent:
                    words.append(word)
            counter = Counter(words)
            word_count = len(words)
            for word in words:
                tf = counter[word]/word_count
                df = self.DF[word] if word in self.vocab else 0
                idf = np.log((no_of_docs+1)/(df + 1))
                tf_idf[doc_id, word] = tf*idf

        return tf_idf

    def __cos_sim(self,x,y):
        if np.amax(y) == 0:
            return 0
        return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))



    def __gen_query_vector(self,query):

        query_vector = np.zeros(len(self.vocab))


        counter = Counter(query)
        word_count = len(query)

        for word in query:
            tf = counter[word]/word_count
            df = self.DF[word] if word in self.vocab else 0
            idf = np.log((self.no_of_docs+1)/(df + 1))
            if word in self.vocab:
                word_index = self.vocab.index(word)
                query_vector[word_index] = tf*idf

        return np.array(query_vector)

    def train(self,docs,docIDs):

        self.no_of_docs = len(docIDs)
        self.__buildDF(docs,docIDs)

        vocab_size = len(self.vocab)
        
        # calculating tf-idf for each term in the vocabulary
        tf_idf = self.__calucateTF_IDF(docs,docIDs)

        self.id2index = {}
        i = 0
        for id in docIDs:
            self.id2index[id] = i
            i += 1
    
        term_doc_matrix = np.zeros((vocab_size,self.no_of_docs))
        

        # creating document index using the tf-idf values
        for i in tf_idf:
            word_index = self.vocab.index(i[1])
            term_doc_matrix[word_index][self.id2index[i[0]]] = tf_idf[i]			

        self.term_doc_matrix = term_doc_matrix

        u, s, v = np.linalg.svd(self.term_doc_matrix)
    
        s = np.diag(s)
        k = self.rank_approximation


        self.u_k,self.s_k,self.v_k = u[:, :k], s[:k, :k], v[:k, :]
        

    def rank(self,queries):
        
        document_vectors = self.v_k.T @ self.s_k

        doc_IDs_ordered = []

        for q in queries:
            query = []
            for s in q:
                for w in s:
                    query.append(w)

            query_vector = self.__gen_query_vector(query)

            pseudo_query_vector = query_vector.T @ self.u_k

            cos_similarities = []
            for i in range(document_vectors.shape[0]):
                cos_similarities.append(
                    self.__cos_sim(pseudo_query_vector, document_vectors[i][:]))

            ids = np.array(cos_similarities).argsort()[-11:][::-1]
            ids = ids + 1

            doc_IDs_ordered.append(ids)

        return doc_IDs_ordered




        
