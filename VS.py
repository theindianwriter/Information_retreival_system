from util import *
import numpy as np
from collections import Counter

# Add your import statements here


class VS():

    def __init__(self):
        self.index = None

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

    def __cos_sim(self, x, y):
        if np.amax(y) == 0:
            return 0
        return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

    def __gen_query_vector(self, query):

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

        return query_vector

    def train(self, docs, docIDs):
        """
                Builds the document index in terms of the document
                IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
                A list of lists of lists where each sub-list is
                a document and each sub-sub-list is a sentence of the document
        arg2 : list
                A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        index = None

        # building the document frequency for each word in the corpus
        self.no_of_docs = len(docIDs)
        self.__buildDF(docs, docIDs)

        vocab_size = len(self.vocab)

        # calculating tf-idf for each term in the vocabulary
        tf_idf = self.__calucateTF_IDF(docs, docIDs)

        index = {}
        for doc_id in docIDs:
            index[doc_id] = np.zeros(vocab_size)

        # creating document index using the tf-idf values
        for i in tf_idf:
            word_index = self.vocab.index(i[1])
            index[i[0]][word_index] = tf_idf[i]

        self.index = index

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
                A list of lists of lists where each sub-list is a query and
                each sub-sub-list is a sentence of the query


        Returns
        -------
        list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        """
        doc_IDs_ordered = []

        for q in queries:
            query = []
            for s in q:
                for w in s:
                    query.append(w)

            query_vector = self.__gen_query_vector(query)

            cos_similarities = []
            for i in self.index:
                cos_similarities.append(
                    self.__cos_sim(query_vector, self.index[i]))

            ids = np.array(cos_similarities).argsort()[-11:][::-1]
            ids = ids + 1

            doc_IDs_ordered.append(ids)

        return doc_IDs_ordered
