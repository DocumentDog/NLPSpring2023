#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:57:08 2023

@author: chenhanchuan
"""

import math
import string
import numpy as np

closed_class_stop_words = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed',\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or'
                           ]
#%% TFIDF score for cran.qry
cran_qry = open('/Users/chenhanchuan/Desktop/NLP/HW4/Cranfield_collection_HW/cran.qry','r')
cran_qry = cran_qry.readlines()


#create a dictionary that contains all queries indexed by number
qry = {}
word_list = []
index = 0
for line in cran_qry:
    line = line.split()
    if line.__contains__('.I'):
        index += 1
        if index not in qry.keys():
            qry[index] = []
        continue
    if line.__contains__('.W'):
        continue
    for word in line:
        word = word.strip(string.punctuation)
        if word not in word_list and word != '':
            word_list.append(word)
        qry[index].append(word)
    
#count number of queries containing the specific word
IDF_qry = {}
for word in word_list:
    for i in qry.keys():
        if word in qry[i] and word not in closed_class_stop_words and word not in string.punctuation and word.isdigit() == False:
            if word not in IDF_qry.keys():
                IDF_qry[word] = 0
            IDF_qry[word] += 1
            continue
        
#calculate the IDF scores for each word in the collection of queries
for key in IDF_qry.keys():
    IDF_qry[key] = math.log(len(qry) / IDF_qry[key])  

            
#count the number of instance of each word in query, which is TF score
TF_qry = {}
for word in word_list:
    if word not in closed_class_stop_words and word.isdigit() == False:
        if word not in TF_qry.keys():
            TF_qry[word] = []
        for i in qry.keys():
            count = qry[i].count(word)
            TF_qry[word].append(count)

for lst in TF_qry.keys():
    TF_qry[lst] = np.array(TF_qry[lst])

#the vector list TFIDF score for words in vector
TFIDF_qry = {}
for i in TF_qry.keys():
    for j in IDF_qry.keys():
        if i == j:
            if i not in TFIDF_qry.keys():
                TFIDF_qry[i] = []
            TFIDF_qry[i] = TF_qry[i] * IDF_qry[j]
            break
        
#%% TFIDF score for abstract
cran_all_1400 = open('/Users/chenhanchuan/Desktop/NLP/HW4/Cranfield_collection_HW/cran.all.1400','r')
cran_all_1400 = cran_all_1400.readlines()

#create a dictionary that contains all 1400 abstract indexed by number
abstract = {}
word_list_abstract= []
index = 0
for i in range(len(cran_all_1400)):
    if cran_all_1400[i].__contains__('.I'):
        index += 1
        if index not in abstract.keys():
            abstract[index] = []
    
    if cran_all_1400[i].__contains__('.W'):
        for j in range(i + 1, len(cran_all_1400)):
            if cran_all_1400[j].__contains__('.I'):
                break
            line = cran_all_1400[j].split()
            for word in line:
                word = word.strip(string.punctuation)
                if word not in word_list_abstract and word != '':
                    word_list_abstract.append(word)
                abstract[index].append(word)

#count number of abstract containing the specific word
IDF_abstract = {}
for word in word_list_abstract:
    for i in abstract.keys():
        if word in abstract[i] and word not in closed_class_stop_words and word[0].isdigit() == False and word.__contains__('-') == False:
            if word not in IDF_abstract.keys():
                IDF_abstract[word] = 0
            IDF_abstract[word] += 1
            continue
        
#calculate the IDF scores for each word in the collection of abstract
for key in IDF_abstract.keys():
    IDF_abstract[key] = math.log(len(abstract) / IDF_abstract[key])  

#count the number of instance of each word in query, which is TF score
TF_abstract = {}
for word in word_list_abstract:
    if word not in closed_class_stop_words and word[0].isdigit() == False and word.__contains__('-') == False:
        if word not in TF_abstract.keys():
            TF_abstract[word] = []
        for i in abstract.keys():
            count = abstract[i].count(word)
            TF_abstract[word].append(count)

for lst in TF_abstract.keys():
    TF_abstract[lst] = np.array(TF_abstract[lst])

#the vector list TFIDF score for words in vector for abstract
TFIDF_abstract = {}
for i in TF_abstract.keys():
    for j in IDF_abstract.keys():
        if i == j:
            if i not in TFIDF_abstract.keys():
                TFIDF_abstract[i] = []
            TFIDF_abstract[i] = TF_abstract[i] * IDF_abstract[j]
            break

#%% Assign TFIDF score of each word to orighinal sentences in quries and abstract
qry_vector = {}
for i in qry.keys():
    if i not in qry_vector.keys():
        qry_vector[i] = []
    for word in qry[i]:
        if word in TFIDF_qry.keys():
            qry_vector[i].append(TFIDF_qry[word][i - 1])
        else:
            qry_vector[i].append(0.0)

for vec in qry_vector.keys():
    qry_vector[vec] = np.array(qry_vector[vec])

qry_final = {}
for i in qry.keys():
    if i not in qry_final.keys():
        qry_final[i] = {}
    count = 0
    for word in qry[i]:
        if word not in qry_final[i].keys():
            qry_final[i][word] = qry_vector[i][count]
        count += 1
        

abstract_vector = {}
for j in abstract.keys():
    if j not in abstract_vector.keys():
        abstract_vector[j] = []
    for word in abstract[j]:
        if word in TFIDF_abstract.keys():
            abstract_vector[j].append(TFIDF_abstract[word][j - 1])
        else:
            abstract_vector[j].append(0.0)
            
for vec in abstract_vector.keys():
    abstract_vector[vec] = np.array(abstract_vector[vec])

abstract_final = {}
for i in abstract.keys():
    if i not in abstract_final.keys():
        abstract_final[i] = {}
    count = 0
    for word in abstract[i]:
        if word not in abstract_final[i].keys():
            abstract_final[i][word] = abstract_vector[i][count]
        count += 1
        
#%% Calculate the cosine similarity between vectors for query and abstract
score = {}
cosim = 0
for i in qry_final.keys():
    sum_qry = 0 
    cur_qry = list(qry_final[i].values())
    if i not in score.keys():
        score[i] = {}
    for m in cur_qry:
        sum_qry += m ** 2
    
    for ab in abstract_final.keys():
        sum_abstract = 0
        if ab not in score[i].keys():
            score[i][ab] = 0
        cur_ab = []
        for word in qry_final[i].keys():
            if word in abstract_final[ab].keys():
                cur_ab.append(abstract_final[ab][word])
            else:
                cur_ab.append(0)
        
        
        for n in cur_ab:
            sum_abstract += n ** 2
            
        numerator = np.dot(cur_qry, cur_ab)
        denominator = math.sqrt(sum_qry * sum_abstract)
        if denominator != 0:
            cosim = numerator / denominator
        score[i][ab] = cosim

#sort the cosine similarity 
for i in score.keys():
    score[i] = {k: v for k, v in sorted(score[i].items(), key=lambda item: item[1], reverse=True)} 

#%% output the sorted cosine similarity value
with open('output.txt', 'w') as f:
    for q in score.keys():
        for ab in score[q].keys():
            f.write('{0}'' ''{1}'' ''{2}\n'.format(str(q), str(ab), str('{:f}'.format(score[q][ab]))))
f.close()








































