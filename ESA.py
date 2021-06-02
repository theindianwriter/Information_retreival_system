
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from util import *
import numpy as np
from collections import Counter
import os

class ESA():

    def __init__(self):
        
        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()


    def __preprocessDocs(self, docs):
        
        # Segment docs
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.sentenceSegmenter.punkt(doc)
            segmentedDocs.append(segmentedDoc)
    
        # Tokenize docs
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenizer.pennTreeBank(doc)
            tokenizedDocs.append(tokenizedDoc)

        # Stem/Lemmatize docs
        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.inflectionReducer.reduce(doc)
            reducedDocs.append(reducedDoc)
        # Remove stopwords from docs
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.stopwordRemover.fromList(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs


    def __load_wiki_data(self):
        
        articles_filename = os.listdir("Wikipedia/")
        wiki_docs = []
        wiki_docsID = []
        id = 1
        for filename in articles_filename:
            with open("Wikipedia/"+filename, 'r') as f:
                data = f.read()

            wiki_docs.append(data)
            wiki_docsID.append(id)
            id += 1

        return wiki_docs,wiki_docsID

    def __calucateTF_IDF(self, docs, docIDs,vocab,DF):

        vocab_size = len(vocab)
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
                df = DF[word] if word in vocab else 0
                idf = np.log((no_of_docs+1)/(df + 1))
                tf_idf[doc_id, word] = tf*idf

        return tf_idf

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

        return vocab,DF

    def __cos_sim(self, x, y):
        if np.amax(x) == 0:
            return 0
        if np.amax(y) == 0:
            return 0
        return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

    def __gen_query_vector(self, query):

        query_vector = np.zeros(len(self.dataset_vocab))

        counter = Counter(query)
        word_count = len(query)

        for word in query:
            tf = counter[word]/word_count
            df = self.dataset_DF[word] if word in self.dataset_vocab else 0
            idf = np.log((self.no_of_docs+1)/(df + 1))
            if word in self.dataset_vocab:
                word_index = self.dataset_vocab.index(word)
                query_vector[word_index] = tf*idf

        return query_vector


    def train(self, docs, docIDs):
      
        dataset_vocab,dataset_DF = self.__buildDF(docs, docIDs)
    
        dataset_tf_idf = self.__calucateTF_IDF(docs, docIDs,dataset_vocab,dataset_DF)


        dataset_vocab_size = len(dataset_vocab)

        dataset_id2index = {}
        i = 0
        for id in docIDs:
            dataset_id2index[id] = i
            i += 1

        dataset_term_doc_matrix = np.zeros((dataset_vocab_size, len(docIDs)))

        # creating document index using the tf-idf values
        for i in dataset_tf_idf:
            word_index = dataset_vocab.index(i[1])
            dataset_term_doc_matrix[word_index][dataset_id2index[i[0]]] = dataset_tf_idf[i]

        self.dataset_term_doc_matrix = dataset_term_doc_matrix
        self.dataset_vocab = dataset_vocab
        self.dataset_DF = dataset_DF
        self.no_of_docs = len(docIDs)


        wiki_docs_raw, wiki_docIDs = self.__load_wiki_data()

        wiki_docs = self.__preprocessDocs(wiki_docs_raw)

        wiki_vocab, wiki_DF = self.__buildDF(wiki_docs, wiki_docIDs)


        wiki_tf_idf = self.__calucateTF_IDF(wiki_docs, wiki_docIDs,wiki_vocab,wiki_DF)

        wiki_vocab_size = len(wiki_vocab)

        wiki_id2index = {}
        i = 0
        for id in wiki_docIDs:
            wiki_id2index[id] = i
            i += 1

        wiki_term_doc_matrix = np.zeros((wiki_vocab_size, len(wiki_docIDs)))

        # creating document index using the tf-idf values
        for i in wiki_tf_idf:
            word_index = wiki_vocab.index(i[1])
            wiki_term_doc_matrix[word_index][wiki_id2index[i[0]]] = wiki_tf_idf[i]

        self.wiki_term_doc_matrix = wiki_term_doc_matrix

        self.no_of_wiki_docs = len(wiki_docIDs)

        dataset_term_in_wiki_index = []
        for word in dataset_vocab:
            if word not in wiki_vocab:
                dataset_term_in_wiki_index.append(-1)
            else:
                word_index = wiki_vocab.index(word)
                dataset_term_in_wiki_index.append(word_index)


        self.dataset_term_in_wiki_index = dataset_term_in_wiki_index

        self.document_vectors = np.zeros((len(docIDs), len(wiki_docIDs)))
        # print(dataset_term_doc_matrix.shape[1])
        for i in range(dataset_term_doc_matrix.shape[1]):
            term_tf_idf_values = self.dataset_term_doc_matrix[:,i]

            for j,val in enumerate(term_tf_idf_values):
                if dataset_term_in_wiki_index[j] >= 0:
                    self.document_vectors[i] += (wiki_term_doc_matrix[dataset_term_in_wiki_index[j]][:]*val)


    def rank(self,queries):
        # for i in range(100):
        #     #print(np.array_equal(self.document_vectors[i], self.document_vectors[i+1]))
        #     print(np.amax(self.document_vectors[i]))
        doc_IDs_ordered = []
        for q in queries:
            query = []
            for s in q:
                for w in s:
                    query.append(w)

            query_vector = self.__gen_query_vector(query)
            pseudo_query_vector = np.zeros(self.no_of_wiki_docs)
            for j, val in enumerate(query_vector):
                if self.dataset_term_in_wiki_index[j] >= 0:
                    pseudo_query_vector += (self.wiki_term_doc_matrix[self.dataset_term_in_wiki_index[j]][:]*val)

            cos_similarities = []
            for i in range(self.document_vectors.shape[0]):
                cos_similarities.append(
                    self.__cos_sim(pseudo_query_vector, self.document_vectors[i][:]))

            ids = np.array(cos_similarities).argsort()[-11:][::-1]
            ids = ids + 1

            doc_IDs_ordered.append(ids)
        
        return doc_IDs_ordered
