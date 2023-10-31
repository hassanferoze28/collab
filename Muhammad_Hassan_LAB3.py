#!/usr/bin/env python
# coding: utf-8

# # IMPORT DICTIONARIES

# In[3]:


pip install -U spacy


# In[4]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[22]:


import spacy


# In[23]:


import en_core_web_sm


# In[24]:


nlp = en_core_web_sm.load()


# In[25]:


from spacy import displacy


# In[26]:


from pprint import pprint


# In[27]:


from collections import Counter


# In[28]:


import nltk
import bs4 as bs
import urllib.request
import re
import pandas as pd


# # (1.1) Scrape Webpage Through Beautiful Soup

# In[29]:


def _scrape_webpage(url):
    scrapped_textdata = urllib.request.urlopen(url)
    textdata = scrapped_textdata.read()
    parsed_textdata = bs.BeautifulSoup(textdata,'lxml')
    paragraphs = parsed_textdata.find_all('p')
    formatted_text = ""
    for para in paragraphs:
        formatted_text += para.text
    return formatted_text  


# In[30]:


mytext = _scrape_webpage('https://www.washingtonpost.com/politics/new-jersey-governor-murphy-ciattarelli/2021/11/03/805db27e-3843-11ec-9bc4-86107e7b0ab1_story.html')


# # (1.2) Sentence Segmentation

# In[120]:


doc = nlp(mytext)

for sent in doc.sents:
        print(sent)
    


# # Get the Token Level Entity Annotations

# In[121]:


pprint([(sent, sent.ent_iob_ , sent.ent_type_) for sent in doc])


# # Get The Named Entities

# In[122]:


pprint([(sent.text, sent.label_) for sent in doc.ents])


# # (1.2.1) Count Every Named Entity

# In[123]:


labels = [sent.label_ for sent in doc.ents]
from collections import Counter
Counter(labels)


# # Count the Total Number Of Named Entities

# In[124]:


entity_freq = Counter(doc.ents)
len(entity_freq)


# # (1.2.2) Count Most Frequent Tokens

# In[206]:


common_entity = entity_freq.most_common(100)
common_entity


# In[126]:


[(sent.orth_,sent.pos_,sent.lemma_) 
 for sent in [x for x in doc if not x.is_stop and x.pos_ != 'PUNT']]


# # (1.2.3) Pick a random integer K using Python random module

# In[127]:


import random
import re


# In[129]:


a = random.randint(0,153)
b = a-1
c = a-2

print(a,b,c)


# In[130]:


sentences = [x for x in doc.sents]
print(sentences[a],sentences[b],sentences[c])


# In[207]:


consec_Sentences = "No Democrat had won reelection for governor since 1977 in New Jersey — and the last two GOP governors both won their first elections in the first year of a new Democratic president. If history were a guide, Ciattarelli was well-positioned to win Tuesday. Republicans have already cited those New Jersey Democrats as top targets in their quest to win back the House and further blunt Biden’s agenda."


# # (1.2.4) Extract part-of-speech and lemmatize these consecutive sentences

# In[208]:


kth = nlp(consec_Sentences)
[(sent.orth_,sent.pos_,sent.lemma_) 
 for sent in [x for x in kth if not x.is_stop and x.pos_ != 'PUNT']]


# # (1.2.5) Get and print the entity annotation for each token of the Kth sentence
# 

# In[209]:


pprint([(sent, sent.ent_iob_ , sent.ent_type_) for sent in kth])


# In[133]:


pprint([(sent.text, sent.label_) for sent in kth.ents])


# # (1.2.6) Visualize the entities

# In[134]:


displacy.render(kth, jupyter = True, style ='ent')


# # Visualize the dependencies of Kth sentence 

# In[226]:


displacy.render(kth, style ='dep', jupyter = True, options = {"DISTANCE": 50})


# # (1.2.7) Visualize all the entities in the document 

# In[135]:


displacy.render(doc, jupyter = True, style ='ent')


# # (2.1) De-identify all person names (PERSON) in the webpage document with [REDACTED]
# and visualize them as shown in class.

# In[36]:


def replace_ner(text):
    clean_text = mytext
    doc = nlp(mytext)
    for ent in reversed(doc.ents):
        
#if label equal to person, then find the location and convert it into Redacted
      if ent.label_ == 'PERSON':
        
        
        clean_text = clean_text[:ent.start_char]+'[REDACTED]' + clean_text[ent.end_char:]
    return clean_text
    


# In[37]:


clean_text = replace_ner('mytext')
displacy.render(nlp(clean_text),jupyter = True, style = 'ent')


# In[ ]:




