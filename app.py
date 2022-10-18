
from fileinput import filename
from importlib.metadata import files
from flask import Flask,request,jsonify,render_template
from joblib import load
import os
import re
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime

stoplist = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()


# create flask app
app=Flask(__name__)

#load pickle model
vector=load("vectors_product.joblib")
model=load("model_product.joblib")


@app.route("/")
def Home():
    return  render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    review_txt = request.form.get("review")

    text = [review_txt]
    vec = vector.transform(text)
    prediction=model.predict(vec)  
    prediction=int(prediction) 

    if(prediction>0):
        prediction="Congratulation ! It is Positive Review"
    else:
        prediction="OMG ! It is Negetive Review"

    return  render_template("index.html",prediction_text=prediction)



@app.route("/graphs", methods=["post"])
def graphs():

    file=request.files['files']
    date_string = datetime.datetime.now().strftime("%H-%M-%S")

    file_name=date_string+file.filename
    filepath=os.path.join('static',file_name)
    file.save(filepath)

    csv_path='static/'+file_name

    data=pd.read_csv(csv_path)
    
    prediction_vec=[]

    postive_sum=0
    negetive_sum=0
    for sentence in data.review.values:
    
        text = [sentence]
        vec = vector.transform(text)
        prediction=model.predict(vec)  
        prediction=int(prediction)

        if (prediction==1):
            postive_sum=postive_sum+1
        else:
            negetive_sum=negetive_sum+1

        prediction_vec.append(prediction)

    date_string = datetime.datetime.now().strftime("%H-%M-%S")
    

    labels=['Positive','Negetive']
    values=[postive_sum,negetive_sum]
    plt.bar(labels,values)
    plt.xlabel("Review Type")
    plt.ylabel("Review Count")
    imagepath=os.path.join('static',date_string+'.png')
    plt.savefig(imagepath)

    return  render_template("index.html",image=imagepath)
    




if __name__=="__main__":
    app.run(debug=True)

