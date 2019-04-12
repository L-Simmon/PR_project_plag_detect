# -*- coding: utf-8 -*-
# @Time    : 2019/4/10 19:29
# @Author  : xilu
# @File    : Preprocessing.py

import re
import nltk
import string
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import jieba

DELETECHARS = ''.join([string.punctuation, string.whitespace])
stpwrdpath = os.getcwd() + "/stop_words/stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'r')
stpwrd_content = stpwrd_dic.read()
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()


# only preprocess one txt
# 对中文而言，标点和停止词放到了一起，所以最好remove_stop_worlds


class Preprocess():
    def __init__(self, txt, mode='cn', remove_stop_worlds=True, split_sents=True):
        self.txt = txt
        self.raw_sentences = []
        self.for_find_sen = []
        self.idx_sen = None
        self.sentences_tokenized = None
        self.is_plags = []
        self.split_s = split_sents
        if mode == 'en':
            self.sentences_tokenized = self.tokenize_for_en(remove_stop_worlds)
        elif mode == 'cn':
            self.sentences_tokenized = self.tokenize_for_cn(remove_stop_worlds, cut_all=False)
        else:
            raise ValueError()
        for s in self.raw_sentences:
            self.is_plags.append(False)

    def tokenize_for_en(self, remove_stop_worlds=False):
        path = 'english.pickle'
        if not os.path.exists(path):
            raise EnvironmentError('the english.pickle does not exit in the path: {}'.format(path))
        sent_detector = nltk.data.load('english.pickle')
        self.raw_sentences.extend(sent_detector.tokenize(self.txt))
        self.idx_sen = sent_detector.span_tokenize(self.txt)
        sentences = sent_detector.tokenize(self.txt.lower())
        sentences_tokenized = []
        for s, sentence in enumerate(sentences):
            # 用于转换. car's -> car
            sentence = re.sub(r"'\w*", "", sentence)
            sentence = re.sub(r'’\w*', '', sentence)
            #  仅保留字母数字和空格,删除标点
            sentence = re.sub(r'([^\s\w]|_)+', ' ', sentence)
            # 删除纯数字
            sentence = re.sub(r'\s[0-9]+\s', " ", sentence)
            # 分词
            tokenizer = RegexpTokenizer(r'\w+')
            sentence_tokens = tokenizer.tokenize(sentence)
            # 词形归并
            wordnet_lemmatizer = WordNetLemmatizer()
            sentence_tokenized = []
            if remove_stop_worlds:
                for word in sentence_tokens:
                    if word not in stopwords.words('english'):
                        sentence_tokenized.append(wordnet_lemmatizer.lemmatize(word))
            else:
                for word in sentence_tokens:
                    sentence_tokenized.append(wordnet_lemmatizer.lemmatize(word))
            if len(sentence_tokenized) >= 1:
                sentences_tokenized.append(sentence_tokenized)
                self.for_find_sen += [s]
        return sentences_tokenized

    def tokenize_for_cn(self, remove_stop_worlds, cut_all=False):
        self.for_find_sen = []
        if self.split_s:
            sentences = re.split('。|！|\!|\.|？|\?', self.txt)
        else:  # For LCMC
            sentences = self.txt
        sentences_tokenized = []
        for s, text in enumerate(sentences):
            if len(text) >= 1:
                if self.split_s:
                    seg_list = jieba.cut(text, cut_all)
                    self.raw_sentences += [text]
                else:
                    seg_list = text
                    self.raw_sentences += [''.join(text)]
                mywordlist = []
                # liststr = list(seg_list)
                # print(liststr)
                if remove_stop_worlds:
                    for word in seg_list:
                        if not (word.strip() in stpwrdlst) and len(word) >= 1:
                            mywordlist.append(word)
                else:
                    mywordlist = seg_list
                if len(mywordlist) >= 1:
                    sentences_tokenized += [mywordlist]
                    self.for_find_sen += [s]
        return sentences_tokenized

