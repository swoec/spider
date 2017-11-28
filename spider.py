# -*- coding: utf-8 -*-
#encoding=utf-8
"""
Created on Mon Nov 27 11:23:29 2017

@author: alex
"""

import sys  
reload(sys)  
sys.setdefaultencoding("utf-8")  

import time  
import random 
import requests 


import re  
from bs4 import BeautifulSoup  
import json  
import threading  
from requests import Session  

class pspider:
    def __init__(self):
        self.url = 'http://www.runoob.com/python/python-object.html'
        
        self.headers={  
            'Content-Type': 'application/x-www-form-urlencoded'    
        } 
        print '-init--'
        
    def start(self):
        self.s=Session()
        self.parsehtml(self.url)
        print 'start--'
        
    
    def parsehtml(self,url):
        #print '---parse--'
        print '----', url
        _json=dict()
        fo = open('/home/alex/example.txt','a+')
        #html = self.s.post(url,headers=self.headers).text
        post=requests.get(url)
        post.encoding = 'utf-8'
        
       # html = unicode(html)
        soup = BeautifulSoup(post.text, 'lxml')
        #print soup
        #for k in soup.find_all('a'):
           #print 'link---',k.get('href')
        
        #for i in soup.find('div',class="container main").find('div',class="row").find('div',class="col left-column").find('div',id="leftcolumn"):
        for i in soup.find_all('div',id="leftcolumn"):
            #print 'link-------',i.get('href')
           #print 'title---',i.find_all(string=re.compile('href'))
           #print i.get_text()
           for k in i.find_all('a'):
                #print k.get('href')
                newurl='http://www.runoob.com/'+k.get('href')
                #print newurl
                #newhtml=self.s.post(newurl,headers=self.headers).text
                newpost=requests.get(newurl)
                newpost.encoding = 'utf-8'
                newsoup=BeautifulSoup(newpost.text,'lxml')
                for arc in newsoup.find_all('div',class_="article-body"):
            
                   for dc in arc.find_all('div',class_="example"):
                         fo.write(dc.get_text())
                         print dc.get_text().encode('utf-8')
                
           
        fo.close()
       
                  
        
class spiderall:
      #global _dic 
      def __init__(self,url):
          self.url = url
          self._dic = dict()
          
      def start(self):
          self.getalllink(self.url)
          print '---'
          
      def getalllink(self,url):
          print 'getalllink---',url
          post=requests.get(url)
          post.encoding = 'utf-8'
          t= pspider()
          t.parsehtml(url)
          #print self._dic.items()
        
          # html = unicode(html)
          soup = BeautifulSoup(post.text, 'lxml')
          
          for le in soup.find_all('a'):
             link = le.get('href')
             
             res = self._dic.get(link)
             print 'res---',res
             if '1' == res:
                 continue
             else:
                 self._dic[link]='1'
                 #print link
                 if re.match('.+\.(mp4|mp3)$',link,re.M|re.I):
                     continue
                 elif re.match(r'//www\.runoob\.com/',link,re.M|re.I):
                     continue
                 #if link.startwith('www.runoob.com') or link.startwith('http://www.runoob.com/') or link.startwith('//www.runoob.com'):
                 elif re.match(r'^www\.runoob\.com*',link,re.M|re.I) or re.match(r'^http://www\.runoob\.com*',link,re.M|re.I) or re.match(r'^//www\.runoob\.com*',link,re.M|re.I):
                     self.getalllink(link)
                     #allpost.encoding = 'utf-8'
                 elif re.match(r'/',link,re.M|re.I) or re.match(r'^java',link,re.M|re.I) :
                     continue
                 else :
                     #print '---'
                     self.getalllink('http://www.runoob.com/'+link)
                     #apost = requests.get('http://www.runoob.com/'+link):
        

if __name__=='__main__':
     url ='http://www.runoob.com/python/python-object.html'
     t=pspider();
     t.start()
     b=spiderall(url);
     b.start()

    
        
        