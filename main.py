# Import Library
# from app import app
import pandas as pd
import numpy as np
import re

import csv
import os
import pandas
import re
import json
import string

from flask import flash, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from bs4 import BeautifulSoup

from flask import Flask, jsonify, request, Response
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import psycopg2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

# app = Flask(__name__)
app=Flask(__name__,
    static_folder = 'app/static',
    template_folder='app/templates')

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:cintaallah123@localhost:5432/final'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# UPLOAD_FOLDER = './upload'
SAVED_FOLDER = './saved'
ALLOWED_EXTENSIONS = set(['csv'])
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAVED_FOLDER'] = SAVED_FOLDER

class dataset_model(db.Model):
    __tablename__ = 'dataset'

    NewsHeadline = db.Column(db.String(), primary_key=True)

    def __init__(self, NewsHeadline):
        # self.Sentiment = Sentiment
        self.NewsHeadline = NewsHeadline

    def __repr__(self):
        return '<NewsHeadline {}>'.format(self.NewsHeadline)

class predict_model(db.Model):
    __tablename__ = 'predict'

    NewsHeadline = db.Column(db.String(), primary_key=True)
    Sentiment = db.Column(db.String())

    def __init__(self, Sentiment, NewsHeadline):
        self.Sentiment = Sentiment
        self.NewsHeadline = NewsHeadline

    def __repr__(self):
        return '<Sentiment {}>'.format(self.Sentiment)

# initial route
@app.route('/')
# def home():
#     return render_template('index.html')

# index route : show rendered template dashboard (index.html)
@app.route('/index')
def index():
    return render_template('index.html')


# preprocessing route
@app.route('/preprocessing')
def preprocessing():
    return render_template('pre-upload.html')

@app.route('/preprocessing/testing', methods = ['GET', 'POST'])
def upload_file_testing():
    if request.method == 'POST':
        file = request.files['file']
        filename = pd.read_csv(file, encoding="latin-1")
        text = filename['News Headline'].tolist()
        for i in text:
            filename = dataset_model(NewsHeadline=i)
            db.session.add(filename)
            db.session.commit()
        return render_template('pre-uploadview.html')

# file process
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def csv_convert_result_pre(fname):
    filename = (os.path.join(app.config['SAVED_FOLDER'], fname))
    table = pandas.read_csv(filename, encoding='latin-1')
    rows = table.shape[0]
    cols = table.shape[1]
    tableID = BeautifulSoup(table.to_html(classes="table table-striped table-bordered", index_names=False, index=False), "html.parser")
    tableID.find('table')['id'] = 'table-testing'
    dataTable = {'Table': tableID, 'Rows': rows, 'Cols': cols}
    return dataTable

@app.route('/process', methods = ['GET', 'POST'])
def process():
    df = pandas.read_csv('all-data.csv', encoding="latin-1")

    """
    Data text yang sudah ada perlu dibrsihkan agar data latih lebih baik dan bisa 
    digunakan untuk tahapan selanjutnya. Dikarenakan data yang dipakai adalah judul
    berita, maka data sudah sedikit baik dan hanya memerlukan sedikit perubahan 
    salah satunya menggunakan regex.
    """

    def regex (content):
        # menghapus simbol, angka, kata hubung
        content = re.sub(r'[^A-Za-z\s\/]' , ' ', content)
        # menghapus multispace (karena setelah dihapus simbol, angka dan kata hubung
        # terdapat banyak multispace)
        content = re.sub(r'\s\s+', '', content)
        # menghapus multispace dibelakang kalimat
        content = re.sub(r'\s+$', '', content)

        return content

    # mengaplikasi cleansing menggunakan regex  
    cleansing_result = []
    for i in df['News Headline']:
        cleansing = regex(i)
        cleansing_result.append(cleansing)   
    df['News Headline'] = cleansing_result

    # mengubah text pada kolom News Headline menjadi lower case
    df['News Headline'] = df['News Headline'].str.lower()

    # mengganti kata negative, neutral dan positive menjadi angka
    df['Sentiment'] = df['Sentiment'].replace("negative",0).replace("neutral",1).replace("positive",2)

    # ambil data kalimat News Headline, ubah jadi array
    X = df['News Headline'].values

    # ambil Sentiment, ubah jadi array
    Y = df['Sentiment'].values

    # transform column Y ke kategorikal data (sesuai kasus)
    Y = np_utils.to_categorical(Y, num_classes=3)

    # maksimum frequensi pada setiap kata
    MAX_WORD_FREQ = 500000

    # maksimum number pada setiap News Headline
    MAX_WORD_SEQ = 250

    # set embedding layer dimension
    EMBEDDING_DIM = 50

    # proses tokenisasi pada text News Headline
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['News Headline'].values)
    word_index = tokenizer.word_index
    # print('%s tokens.' % len(word_index))

    # membuat embedding layer dengan glove 
    embeddings_index = {}
    with open('glove.6B.50d.txt',encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    # print("Found %s words." % len(embeddings_index))

    found = 0

    # panjang token plus zero padding
    TOKEN_NUM = len(word_index)+1 

    # mempersiapkan embedding matrix. akan menghasilkan value 0 jika tidak menemukan
    # kata
    embedding_matrix = np.zeros((TOKEN_NUM, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found += 1

    # print("Found {} words from {} ".format(found,len(word_index)))

    # set input dari model
    X_train = tokenizer.texts_to_sequences(df['News Headline'].values)
    X_train = pad_sequences(X_train, maxlen=MAX_WORD_SEQ)
    # print('Shape of data tensor:', X_train.shape)

    # set Sentiment dari model
    Y_train = pd.get_dummies(df['Sentiment']).values
    # print('Shape of label tensor:', Y_train.shape)

    """
    dikarenakan jumlah kelas yang tidak balance maka dari itu digunakan random 
    over sampler, penerapan random over sampler pada kasus ini lebih baik 
    dibandingkan dengan teknik smote  
    """
    ros = RandomOverSampler(random_state=777)
    X_ROS, y_ROS = ros.fit_sample(X_train, Y_train)

    #split data dengan data test sebanyak 20% dari keseluruhan data
    X_train, x_test, Y_train, y_test = train_test_split(X_ROS,y_ROS,test_size=0.2,random_state=42)

    # embedding layer untuk input LSTM
    embedding_layer = Embedding(TOKEN_NUM, EMBEDDING_DIM, weights=[embedding_matrix], input_length = 250, trainable=False)

    # inisiasi dimensi pada embedding layer
    embedding_dim = 50

    """
    GRU merupakan algoritma Neural Network yang kompleks dan sangat baik dalam pengolahan NLP, 
    algoritma ini lebih cepat dalam melakukan training dibandingkan dengan LSTM namun performanya tetap baik,
    GRU menangani masalah kehilangan informasi akibat data sequential yang teralu panjang yang dapat menurunkan hasil training,
    data train yang digunakan tidak terlalu besar maka dari itu GRU cocok dengan kasus ini.
    """
    # inisiasi model sekuensial
    model = Sequential()
    model.add(embedding_layer)

    # menggunakan model GRU yaitu bagian dari RNN yang lebih kompleks
    model.add(GRU(256, dropout=0.25))

    # inisiasi dense layer
    model.add(Dense(64, activation='relu'))

    # inisiasi dense layer. output sebanyak kelas menggunakan softmax untuk kasus
    # multiclass classification
    model.add(Dense(3, activation='softmax'))

    # Compile model menggunakan optimizer adam dengan loss kategorikal
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Lihat summary dari model
    model.summary()
    # fit model 
    # history = model.fit(X_train, Y_train,epochs=100, validation_split=0.2, batch_size=100)

    # melihat hasil akurasi testing
    # result = model.evaluate(x_test,y_test)

    # menyimpan model
    # model.save_weights("Sentiment_Financial_News.h5") 
    #load model
    model.load_weights('Sentiment_Financial_News_.h5')
    
    # uji model pada data baru dengan proses preprocessing yang serupa dengan 
    # training model

    # con=psycopg2.connect(host = 'localhost', database='final', user='postgres', password = 'cintaallah123')
    # cur = con.cursor()
    # cur.execute('select * from dataset')
    # rows = cur.fetchall()

    # rows = rows['News Headline'].tolist()

    # cur.close()
    # con.close()

    test = pandas.read_csv('test.csv', encoding='latin-1')

    def regex (content):

        # menghapus simbol, angka, kata hubung
        content = re.sub(r'[^A-Za-z\s\/]' , ' ', content)

        # menghapus multispace
        content = re.sub(r'\s\s+', '', content)
        
        # menghapus multispace dibelakang kalimat
        content = re.sub(r'\s+$', '', content)

        return content

    cleansing_result = []
    for i in test['News Headline']:
        cleansing = regex(i)
        cleansing_result.append(cleansing)   

    test['News Headline'] = cleansing_result
    # mengaplikasi cleansing menggunakan regex  
    test['News Headline'] = test['News Headline'].str.lower()

    # uji testing menggunakan model yang sudah dibuat
    new_data= test["News Headline"]
    seq = tokenizer.texts_to_sequences(new_data)
    padded = pad_sequences(seq, maxlen=250)
    pred = model.predict(padded)
    labels = ["Negative","Neutral","Positive"]
    # print(pred, labels[np.argmax(pred)])
    # looping untuk memprediksi setiap text
    newtest =[]
    for x in pred:
      newtest.append(labels[np.argmax(x)])
      label = pd.DataFrame(data=newtest,columns=['Sentiment'])
    hasil = pd.concat([test,label], axis=1)
    # menyimpan hasil prediksi 
    hasil = hasil.to_csv(os.path.join(app.config['SAVED_FOLDER'], 'hasil.csv'), index=False)

    filename_new = 'hasil.csv'
    # filename_new = predict_model(Sentiment=Sentiment)
    # db.session.add(filename_new)
    # db.session.commit()
    dataTable = csv_convert_result_pre(filename_new)
    # return render_template('process-done.html')
    return render_template('process-done.html', tableTesting = dataTable['Table'],\
        rows = dataTable['Rows'], cols = dataTable['Cols'],\
        filename = filename_new, dnTesting = False)

if __name__ == '__main__':
	app.run(debug=True)