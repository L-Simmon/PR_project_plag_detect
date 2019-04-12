# PR_project_plag_detect
模式识别大作业简单查重实现，对一系列**中文**文本，查询其与语料库[[LCMC]](http://ota.ox.ac.uk/scripts/download.php?otaid=2474)的相似度情况。

## Requirments
python 3.6

numpy

gensim

jieba (if use Chinese corpus)

~~nltk (if use English corpus)~~

## Usage
- download any pre-trained Chinese word vectors from here: [[Embedding/Chinese-Word-Vectors: 100+ Chinese Word Vectors]](https://github.com/Embedding/Chinese-Word-Vectors)
- download LCMC corpus from here: [[The Lancaster Corpus of Mandarin Chinese]](http://ota.ox.ac.uk/scripts/download.php?otaid=2474)
- run the code below for more information:

```shell
python run_and_test.py -h
```

## Reminder

### Print language

The language for screen output is **Chinese**.

### Recommended threshold

--pre_train: ```--thre 0.7```

--no-pre_train: ```--thre 0.9```
