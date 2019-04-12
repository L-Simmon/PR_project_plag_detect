# -*- coding: utf-8 -*-
# @Time    : 2019/4/10 19:30
# @Author  : xilu
# @File    : run_and_test.py

from Similarity import *
import codecs
import argparse
from xml.etree import ElementTree as ET
from gensim.models import word2vec
from gensim.models import KeyedVectors

parser = argparse.ArgumentParser()
parser.add_argument("--test_file", default='D:/test.txt', type=str, help="the txt file for testing")
parser.add_argument("--test_file_list", default=None, type=str, help="the txt file list for testing")
parser.add_argument("--save_file_path", default=None, type=str, help="the path of saving all similarities")
parser.add_argument("--LCMC_path", default='D:/mark_down/PR_homework/Final_project/2474/2474/Lcmc/data/character',
                    type=str, help="the path of original LCMC corpus")
# parser.add_argument("--is_pre_train", default=True, type=bool, help='using the pre_trained w2v or not.')

flag_parser = parser.add_mutually_exclusive_group(required=False)
flag_parser.add_argument('--pre_train', dest='flag', action='store_true')
flag_parser.add_argument('--no-pre_train', dest='flag', action='store_false')
parser.set_defaults(flag=True)

parser.add_argument("--pre_train_txt", default='D:/mark_down/PR_homework/Final_project/sgns.merge.word/sgns.merge.word',
                    type=str, help="the pre_trained word vectors")
parser.add_argument("--thre", default=0.7, type=float, help="the threshold of the sentence similarity")
args = parser.parse_args()
if args.test_file_list is None:
    if not os.path.exists(args.test_file):
        raise ValueError("测试文件路径 '{}' 不存在！".format(args.test_file))
if not os.path.exists(args.LCMC_path):
    raise ValueError("对比语料库路径 '{}' 不存在！".format(args.LCMC_path))

file_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'j', 'K', 'L', 'M', 'N', 'P', 'R']
all_file_sen = []  # [file][doc][sen][word]
whole_file_sens = []

for f in file_list:
    file_name = 'LCMC_' + f + '.XML'
    file_path = os.path.join(args.LCMC_path, file_name)
    files = ET.parse(file_path).getroot()[1]
    one_file_sen = []
    for file in files:
        sen_dom = file.findall('p/s')
        all_sen = []
        for sen in sen_dom:
            words = sen.findall('w')
            all_w = []
            for word in words:
                w = word.text
                all_w += [w]
            all_sen += [all_w]
            whole_file_sens += [all_w]
        one_file_sen += [all_sen]
    all_file_sen += [one_file_sen]

if args.test_file_list is not None:
    if args.save_file_path is not None:
        if not os.path.exists(args.save_file_path):
            raise ValueError("保存文件路径 '{}' 不存在！".format(args.save_file_path))
        else:
            with open(args.save_file_path, mode='a') as sa:
                sa.write('---{:*^50s}----\n'.format('结果'))
                sa.write('对比数据库：{} \n'.format('LCMC'))

    if not os.path.exists(args.test_file_list):
        raise ValueError("测试文件列表路径 '{}' 不存在！".format(args.test_file_list))
    else:
        with open(args.test_file_list, mode='r') as f:
            all_test_file = f.readlines()

        for file_path in all_test_file:
            if not os.path.exists(file_path.strip()):
                # print(file_path)
                raise ValueError('请将测试的文件都移到当前文件夹 或 在列表中写绝对路径！')

        if args.flag:
            print('读取中...')
            wv_from_text = KeyedVectors.load_word2vec_format(args.pre_train_txt, binary=False)
            print('完毕\n')
            S = Similarity(wv_from_text)
        else:
            print('训练中...')
            model = word2vec.Word2Vec(whole_file_sens, hs=1, min_count=1, window=3, size=200)
            print('完毕\n')
            wv_simple = model.wv
            del model
            S = Similarity(wv_simple)

        for file_path in all_test_file:
            f = codecs.open(file_path.strip())
            txt = f.read()
            P = Preprocess(txt, mode='cn')
            # S = Similarity(wv_from_text)
            print('---{:*^50s}----\n'.format(' 正在对比分析文档 <{}> '.format(file_path.strip())))
            for f, file in enumerate(all_file_sen):
                file_name = 'LCMC_' + file_list[f] + '.XML'
                for d, doc in enumerate(file):
                    doc_name = file_name + '/{}{:02d}'.format(file_list[f], d + 1)
                    P_cor = Preprocess(doc, mode='cn', split_sents=False)
                    S.similarity_doc2corpus_file(P, file_path.strip(), P_cor, doc_name, args.thre)
            num_plag = sum(P.is_plags)
            num_sen = len(P.is_plags)
            print('\n 抄袭句子数目 : 句子总数 = {:d} : {:d}\n'.format(num_plag, num_sen))
            print('重复率：{:.2f}\n'.format(1.0 * num_plag / num_sen))
            print('---{:*^50s}----\n'.format(' 对比分析文档 <{}>完成 '.format(args.test_file)))
            if args.save_file_path is not None:
                with open(args.save_file_path, mode='a') as sa:
                    sa.write('{} 全文重复率：{:.2f}\n'.format(file_path.strip(), 1.0 * num_plag / num_sen))

else:
    f = codecs.open(args.test_file)
    txt = f.read()

    P = Preprocess(txt, mode='cn')
    # print(P.sentences_tokenized)
    # os._exit(0)

    if args.flag:
        print('读取中...')
        wv_from_text = KeyedVectors.load_word2vec_format(args.pre_train_txt, binary=False)
        print('完毕\n')
        S = Similarity(wv_from_text)
    else:
        print('训练中...')
        model = word2vec.Word2Vec(whole_file_sens, hs=1, min_count=1, window=3, size=200)
        print('完毕\n')
        wv_simple = model.wv
        del model
        S = Similarity(wv_simple)

    print('---{:*^50s}----\n'.format(' 正在对比分析文档 <{}> '.format(args.test_file)))
    for f, file in enumerate(all_file_sen):
        file_name = 'LCMC_' + file_list[f] + '.XML'
        for d, doc in enumerate(file):
            doc_name = file_name + '/{}{:02d}'.format(file_list[f], d + 1)
            P_cor = Preprocess(doc, mode='cn', split_sents=False)
            S.similarity_doc2corpus_file(P, args.test_file, P_cor, doc_name, args.thre)
    num_plag = sum(P.is_plags)
    num_sen = len(P.is_plags)
    print('\n 抄袭句子数目 : 句子总数 = {:d} : {:d}\n'.format(num_plag, num_sen))
    print('重复率：{:.2f}\n'.format(1.0 * num_plag / num_sen))
    print('---{:*^50s}----\n'.format(' 对比分析文档 <{}> 完成 '.format(args.test_file)))
