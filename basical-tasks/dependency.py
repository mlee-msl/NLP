# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:32:21 2017

@author: MLee
"""

import pyltp
import os

MODEL_DIR = 'D:nlp/ltp_data_v3.4.0'

#sent = "刘能参加了这次会议。" # 待处理语句
#sent = "我今天特别的开心呀。" # 待处理语句
#sent = '国务院总理李克强调研上海外高桥时提出，支持上海积极探索新机制。'
sent = '让小王到北京学习'

segmentor = pyltp.Segmentor()
segmentor.load(os.path.join(MODEL_DIR, 'cws.model'))
word_list = list(segmentor.segment(sent))
segmentor.release()

postagger = pyltp.Postagger()
postagger.load(os.path.join(MODEL_DIR, 'pos.model'))
pos_list = list(postagger.postag(word_list))
postagger.release()

recognizer = pyltp.NamedEntityRecognizer()
recognizer.load(os.path.join(MODEL_DIR, 'ner.model'))
ne_list = list(recognizer.recognize(word_list, pos_list))
recognizer.release()

parser = pyltp.Parser()
parser.load(os.path.join(MODEL_DIR, 'parser.model'))
arc_list = list(parser.parse(word_list, pos_list))
print("ID", "word", "POS", "NE", "parent", 'dependency', sep='\t')
for i in range(len(arc_list)):
    print(i, word_list[i], pos_list[i], ne_list[i], arc_list[i].head-1, arc_list[i].relation, sep="\t")
parser.release()


