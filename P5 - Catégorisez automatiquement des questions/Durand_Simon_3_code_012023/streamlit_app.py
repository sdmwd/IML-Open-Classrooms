#!/usr/bin/env python
# coding: utf-8

# In[2]:


# pip install streamlit


# In[8]:


import os
import dill
import streamlit as st
import re
import spacy
import joblib
import html5lib
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
# nltk.download('omw-1.4')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
spacy.load('en_core_web_sm')


# In[ ]:


# load the saved CountVectorizer
vectorizer_loaded = joblib.load('D:/Mega/Z_Simon/5 - WORK/1 - Projets/Projet 5/saved_ressources/countvectorizer.joblib')

# load the saved classifier
clf_loaded = joblib.load('D:/Mega/Z_Simon/5 - WORK/1 - Projets/Projet 5/saved_ressources/sgdc_classifier.pkl')

# load the saved MultiLabelBinarizer
mlb_loaded = joblib.load('D:/Mega/Z_Simon/5 - WORK/1 - Projets/Projet 5/saved_ressources/multilabelbinarizer.joblib')


# In[4]:


st.title("Générateur de tags")
subtitle = '<p style="font-size: 30px;">Projet 5 - OpenClassrooms Parcours IML</p>'
st.markdown(subtitle, unsafe_allow_html=True)
sentences = st.text_input('Collez ici un post de Stack Overflow')


# In[7]:


with open('D:/Mega/Z_Simon/5 - WORK/1 - Projets/Projet 5/saved_ressources/mypipeline.pkl', 'rb') as file:
    text_clf = dill.load(file)


# In[ ]:


output = text_clf.predict(sentences)


# In[ ]:


tags = mlb_loaded.inverse_transform(output)


# In[5]:


# Display the value
st.write('Tags suggérés :', tags)


# In[ ]:


# streamlit run D:/Mega/Z_Simon/5 - WORK/1 - Projets/Projet 5/saved_ressources/streamlit_app.py


# In[ ]:


# Bonjour python code python mahine learning java javascript

