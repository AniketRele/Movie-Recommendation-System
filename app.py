from flask import Flask, render_template, request
import pickle
import json
import urllib.request
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


final = pd.read_csv("dataset/final_dataset.csv")
cv = CountVectorizer()
final_cv = cv.fit_transform(final['combined'])
similarity = cosine_similarity(final_cv)

def rcmd(m):
    m = m.lower()
    if m not in final['movie_title'].unique():
        s=['Sorry! The movie you requested is not in our database.', 'Please check the spelling or try with some other movies']
        return(s)
    else:
        i = final.loc[final['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:6] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(final['movie_title'][a])
        return l


app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/recommend',methods=['POST'])
def recommend():
    if request.method == 'POST':
        movie_name = request.form['movie_title']
        predictions = rcmd(movie_name)
        predss=[]
        #gg=[]
        for i in range(len(predictions)):
            predss.append(predictions[i].capitalize())
            #gg = '\n'.join(predss)
    return render_template('recommend.html', prediction=predss)

if __name__=='__main__':
    app.run(debug=True)
