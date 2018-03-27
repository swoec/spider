# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 15:08:09 2018

@author: alex
"""

import os  
import re  
import jieba as ws  
import pandas as pd  
from gensim import models,corpora  
import logging  
      
      
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  
documents=[]  
labels=[]  
class_dir=os.listdir('/home/alex/')  
      
    #读取语料库  
for i in class_dir:  
    if i.endswith(".txt"):  
        currentpath='/home/alex/'
#        print(currentpath)  
        files=os.listdir(currentpath)  
        for f in files:  
          if f.endswith(".txt"):
            tmp_list=[]    
            tmp_str=''  
            try:              
                #print(f)
                file=open(currentpath+f)  
                file_str=file.read()  
                #print(file_str)
                file_str=re.sub('(\u3000)|(\x00)|(nbsp)','',file_str)#正则处理，去掉一些噪音  
                doc=''.join(re.sub('[\d\W]','',file_str))  
                tmp_str='|'.join(ws.cut(doc))  
                tmp_list=tmp_str.split('|')  
                labels+=[i]  
                file.close() 
            except:  
                print('read error: '+currentpath+f)  
            documents.append(tmp_list)  
           
  
               
#------------------------------------------------------------------------------  
#LSI model: latent semantic indexing model  
#------------------------------------------------------------------------------  
#https://en.wikipedia.org/wiki/Latent_semantic_analysis  
#http://radimrehurek.com/gensim/wiki.html#latent-semantic-analysis  
dictionary=corpora.Dictionary(documents)
corpus=[dictionary.doc2bow(doc1) for doc1 in documents]#generate the corpus  
tf_idf=models.TfidfModel(corpus)#the constructor  
      
    #this may convert the docs into the TF-IDF space.  
    #Here will convert all docs to TFIDF  
corpus_tfidf=tf_idf[corpus]  
      
    #train the lsi model  
lsi=models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=9) 
topics=lsi.show_topics(num_words=5,log=0)  
for tpc in topics:  
    print(tpc)  