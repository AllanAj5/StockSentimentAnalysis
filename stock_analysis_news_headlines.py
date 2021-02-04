# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 20:35:47 2021

@author: Adrien
"""

import pandas as pd

df = pd.read_csv('C:/Users/Adrien/.spyder-py3/Allan Python Files/NLP/Stock Sentiment Analysis/Data.csv',
                 encoding='ISO8859-1')

# Check Data Distribution
""" Checking the the amount of DataPoints
dates = df.iloc[:,0]
import datetime
dates['Date1'] = pd.to_datetime(dates ,format='%Y-%m-%d')
ss = (dates['Date1'].dt.year)
print(ss)
sd = ss.value_counts()
sd = (sd.sort_index(ascending=True))
"""
# Train Test Split
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

##  PreProcessing DATA starts

# Data Cleaning
data = train.iloc[:,2:]
data.replace("[^a-zA-Z]"," ",regex=True,inplace=True)

# Renaming Cols Names
num_list = [i for i in range(1,26)]
str_list = [str(i) for i in num_list]
data.columns = str_list             #Try Dynamic Naming in next version
                        
# column wise Case conversion - str.lower()
for i in str_list:
    data[i] = data[i].str.lower()
    
# Joining all strings of a row to make it a para

# Concatenation Col values for each row -  testing = " ".join(str(x) for x in data.iloc[1,:]) -  
corpus = []
for row in range(0,len(data.index)):
    corpus.append(' '.join(str(x) for x in data.iloc[row,:]))



## END of PreProcessing

#Vector Conversion

from sklearn.feature_extraction.text import CountVectorizer
BOW_vectorizer = CountVectorizer(ngram_range=(2,2))
X = BOW_vectorizer.fit_transform(corpus)
y = train['Label']


#------------test data---preproc repeat-----#
test_data = test.iloc[:,2:]
test_data.replace("[^a-zA-Z]"," ",regex=True,inplace=True)

# for fun we are renaming col names
fun_list = [str(i) for i in range(1,26)]
test_data.columns = fun_list
y_test = test['Label']

for col in test_data.columns:
    test_data[col] = test_data[col].str.lower()
    
test_corpus = []

for row in range(0,len(test_data.index)):
    test_corpus.append(" ".join(str(sent) for sent in test_data.iloc[row,:]))
    
X_test = BOW_vectorizer.transform(test_corpus)

# Training the Model

from sklearn.ensemble import RandomForestClassifier
rf_class5r_model = RandomForestClassifier(n_estimators=200,criterion='entropy')
rf_class5r_model.fit(X,y)
prediction = rf_class5r_model.predict(X_test)


# Moment of Truth - Metrics

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
confusion = confusion_matrix(y_test,prediction)
accuracy = accuracy_score(y_test,prediction)
class_rep = classification_report(y_test,prediction)
print(confusion)
print(accuracy)
print(class_rep)