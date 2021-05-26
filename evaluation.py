from util import *
import numpy as np
from math import log2

# Add your import statements here




class Evaluation():


	def __intersection(self,list1,list2):

		list3 = [value for value in list1 if value in list2]
		return len(list3)

	def __getRelevanceAndPositionList(self,query_ids,qrels):


		ground_truth = { "position": [],"relevance" : []}


		qrels_index = 0
		len_of_qrels = len(qrels)
		for q_id in query_ids:
			pos = []
			rel = []
			while(qrels_index < len_of_qrels):

				if int(qrels[qrels_index]["query_num"]) == q_id:
					pos.append(int(qrels[qrels_index]["position"]))
					rel.append(int(qrels[qrels_index]["id"]))
					qrels_index += 1
				else:
					ground_truth["position"].append(pos)
					ground_truth["relevance"].append(rel)
					break
		
		return ground_truth



	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		numerator = self.__intersection(query_doc_IDs_ordered[:k],true_doc_IDs)
		denominator = k
		precision = numerator/denominator

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		ground_truth = self.__getRelevanceAndPositionList(query_ids,qrels)
		
		precision_list = []
		for query_id in query_ids:
			for query_doc_IDs_ordered,true_doc_IDs in zip(doc_IDs_ordered,ground_truth["relevance"]):
				precision_list.append(self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k))

		meanPrecision = np.mean(np.array(precision_list))
		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		numerator = self.__intersection(query_doc_IDs_ordered[:k],true_doc_IDs)
		denominator = len(true_doc_IDs)
		recall = numerator/denominator

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		ground_truth = self.__getRelevanceAndPositionList(query_ids,qrels)
		
		recall_list = []
		for query_id in query_ids:
			for query_doc_IDs_ordered,true_doc_IDs in zip(doc_IDs_ordered,ground_truth["relevance"]):
				recall_list.append(self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k))

		meanRecall = np.mean(np.array(recall_list))

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if recall > 0 or precision > 0:
			fscore = 2*recall*precision/(recall + precision)
		else:
			fscore = 0

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		ground_truth = self.__getRelevanceAndPositionList(query_ids,qrels)
		
		fscore_list = []
		for query_id in query_ids:
			for query_doc_IDs_ordered,true_doc_IDs in zip(doc_IDs_ordered,ground_truth["relevance"]):
				fscore_list.append(self.queryFscore(query_doc_IDs_ordered, query_id, true_doc_IDs, k))

		meanFscore = np.mean(np.array(fscore_list))

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here
		relevance_score = []
		for position in true_doc_IDs["position"]:
			relevance_score.append(5-position)
		DCG = 0
		for i in range(1,k+1):
			if query_doc_IDs_ordered[i-1] in true_doc_IDs["relevance"]:
				j = true_doc_IDs["relevance"].index(query_doc_IDs_ordered[i-1])
				DCG += relevance_score[j]/log2(i+1)

		iDCG = 0

		relevance_score.sort()
		m = min(k,len(relevance_score))
		for i in range(1,m+1):
			iDCG += relevance_score[-i]/log2(i+1)

		nDCG = DCG/iDCG


		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		ground_truth = self.__getRelevanceAndPositionList(query_ids,qrels)
		
		nDCG_list = []
		for query_id in query_ids:
			for query_doc_IDs_ordered,rel,pos in zip(doc_IDs_ordered,ground_truth["relevance"],ground_truth["position"]):
				true_doc_IDs = {"relevance": rel,"position": pos}
				nDCG_list.append(self.queryNDCG(query_doc_IDs_ordered, query_id, true_doc_IDs, k))

		meanNDCG = np.mean(np.array(nDCG_list))

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = 0
		count = 0
		for i in range(1,k+1):
			if query_doc_IDs_ordered[i-1] in true_doc_IDs:
				avgPrecision += self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i)
				count += 1
		if count:
			avgPrecision = avgPrecision/(count)
		else:
			avgPrecision = 0


		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		ground_truth = self.__getRelevanceAndPositionList(query_ids,q_rels)
		
		avg_precision_list = []
		for query_id in query_ids:
			for query_doc_IDs_ordered,true_doc_IDs in zip(doc_IDs_ordered,ground_truth["relevance"]):
				avg_precision_list.append(self.queryAveragePrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k))

		meanAveragePrecision = np.mean(np.array(avg_precision_list))

		return meanAveragePrecision

