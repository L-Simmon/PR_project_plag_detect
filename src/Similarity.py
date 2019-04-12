# -*- coding: utf-8 -*-
# @Time    : 2019/4/10 19:29
# @Author  : xilu
# @File    : Similarity.py

from Preprocessing import *
import numpy as np


class Similarity():
    def __init__(self, model):
        self.similarity_fun = self.alzahrani_similarity
        self.model = model

    def alzahrani_similarity(self, s1, s2):
        string_similarity = []
        for ws1 in s1:
            max_word_similarity = 0
            for ws2 in s2:
                if (ws1 in self.model and ws2 in self.model):
                    similarity = self.model.similarity(ws1, ws2)
                    if similarity > max_word_similarity:
                        max_word_similarity = similarity
            if max_word_similarity > 0:
                string_similarity.append(max_word_similarity)
        if len(string_similarity) == 0:
            return 0
        return sum(string_similarity) / len(string_similarity)

    def cos_dis(self, a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return np.dot(a, b) / (norm_a * norm_b)

    def just_ave_similarity(self, s1, s2):
        vec_sum_s1 = []
        vec_sum_s2 = []
        dim = self.model.vectors.shape[1]

        for ws1 in s1:
            if ws1 in self.model:
                vec_ws1 = self.model[ws1]
            else:
                vec_ws1 = np.random.rand(dim)
            vec_sum_s1 += [vec_ws1]

        for ws2 in s2:
            if ws2 in self.model:
                vec_ws2 = self.model[ws2]
            else:
                vec_ws2 = np.random.rand(dim)
            vec_sum_s2 += [vec_ws2]

        if len(vec_sum_s2) == 0:
            vec_mean_s2 = np.random.rand(dim)
        else:
            vec_mean_s2 = np.mean(vec_sum_s2, 0)

        if len(vec_sum_s1) == 0:
            vec_mean_s1 = np.random.rand(dim)
        else:
            vec_mean_s1 = np.mean(vec_sum_s1, 0)
        # print(vec_mean_s1.shape, vec_mean_s2.shape)
        return self.cos_dis(vec_mean_s1, vec_mean_s2)

    def similarity_doc2doc(self, doc1, doc2, thre=0.5, language='cn'):
        P1 = Preprocess(doc1, language)
        P2 = Preprocess(doc2, language)
        sents1 = P1.sentences_tokenized
        sents2 = P2.sentences_tokenized
        for i, s1 in enumerate(sents1):
            for j, s2 in enumerate(sents2):
                cos1 = self.similarity_fun(s1, s2)
                cos2 = self.similarity_fun(s2, s1)
                cos = (cos1 + cos2) / 2
                if cos > thre:
                    P1.is_plags[i] = True
                    P2.is_plags[j] = True
        return P1, P2

    def similarity_doc2corpus_file(self, P_doc, test_name, P_cor, corpus_file_name, thre=0.5, language='cn',
                                   verbose=True):
        '''
            待测文档需含标点，没标点不叫文档
        '''
        # P_doc = Preprocess(doc, language)
        sents_doc = P_doc.sentences_tokenized
        sents_corpus = P_cor.sentences_tokenized
        # print('---{:*^50s}----'.format(' 正在分析文档 <{}> '.format(doc_name)))
        for i, s1 in enumerate(sents_doc):
            for j, s2 in enumerate(sents_corpus):
                if len(s2) >= 5:
                    cos1 = self.similarity_fun(s1, s2)
                    cos2 = self.similarity_fun(s2, s1)
                    cos = (cos1 + cos2) / 2
                    # cos = cos1
                    if cos > thre:
                        P_doc.is_plags[i] = True
                        if verbose:
                            print('文档 <{}> 的第 {:d} 句抄自 <{}> 的第 {:d} 句\n'.format(test_name, P_doc.for_find_sen[i] + 1, corpus_file_name,
                                                                                P_cor.for_find_sen[j] + 1))
                            print('---{:*^30s}----\n'.format(' 原句（不含标点） '))
                            print(''.join(P_cor.raw_sentences[P_cor.for_find_sen[j]]) + '\n')
                            print('---{:*^35s}----\n'.format(' 抄成了 '))
                            print(''.join(P_doc.raw_sentences[P_doc.for_find_sen[i]]) + '\n')
                            print('---{:*^38s}----\n'.format(''))
                            print('句子相似度 {:.2f}\n'.format(cos))
        return 0
