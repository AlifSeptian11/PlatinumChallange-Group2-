#import library
from flask import Flask, jsonify, request
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
import pandas as pd
import pickle, re
import sqlite3 as sq
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

class CustomFlaskAppWithEncoder(Flask):
    json_provider_class = LazyJSONEncoder

#memanggil flask objek dan menyimpannya via variable app
app = CustomFlaskAppWithEncoder(__name__)

# menuliskan judul dan input host lazystring untuk random url
swagger_template = dict(
    info = {
        'title' : LazyString(lambda: "API Documentation for Processing and Cleansing"),
        'version' : LazyString(lambda: "1.0.0"),
        'description' : LazyString(lambda: "**Dokumentasi API untuk Processing dan Cleansing Data** \n API BY ALIF"),
    },
    host = LazyString(lambda: request.host)
)

# mendefinisikan endpoint 
swagger_config = {
    "headers" : [],
    "specs" : [
        {
            "endpoint": "docs",
            "route" : "/docs.json",
        }
    ],
    "static_url_path": "/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template, config = swagger_config)

max_features = 5000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
sentiment = ['negative', 'neutral', 'positive']

def lowercase(s):
    return s.lower()

def punctuation(s):
    s = re.sub(r'(?:\@|http?\://|https?\://|www)\S+', '', s) #menghapus https dan http
    s = re.sub('<.*?>', ' ', s) #mengganti karakter html dengan tanda petik
    s = re.sub('[^0-9a-zA-Z]+', ' ', s) #menghilangkan semua karakter yang bukan huruf atau angka dan menggantinya dengan spasi.
    s = re.sub('\n',' ',s) #mengganti line baru dengan spasi
    s = re.sub(r':', ' ', s) #menggantikan karakter : dengan spasi 
    s = re.sub('gue','saya', s) # Mengganti kata "gue" dengan kata "saya"
    s = re.sub(r'\b[a-zA-Z]\b', ' ', s) #menghapus single char
    s = ' '.join(s.split()) #memisahkan dan menggabungkan kata
    s = s.strip() #menghilangkan whitespace di awal dan di akhir teks
    s = re.sub(r'pic.twitter.com.[\w]+', '', s) #menghapus link picture
    s = re.sub(r'\buser\b',' ', s) #menghapus kata 'user'
    s = re.sub(r'\brt\b',' ', s) #menghapus awalan rt
    s = re.sub('RT',' ', s) #menghapus RT simbol
    s = re.sub(r'‚Ä¶', '', s)
    
    return s

# database
def alay_to_normal(s):
    for word in kamusalay:
        return ' '.join([kamusalay[word] if word in kamusalay else word for word in s.split(' ')])
    
def cleansing(sent):
    string = lowercase(sent)
    string = punctuation(string)
    string = alay_to_normal(string)

    return string

conn = sq.connect('database_pl.db', check_same_thread = False)
df_kamusalay = pd.read_sql_query('SELECT * FROM kamusalay', conn)
kamusalay = dict(zip(df_kamusalay['alay'], df_kamusalay['normal']))

# load file sequences CNN
file_CNN = open('CNN/x_pad_sequences.pickle', 'rb')
feature_file_from_cnn = pickle.load(file_CNN)
file_CNN.close()

model_file_from_cnn = load_model('CNN/model.h5')

# load file seuences RNN
file_RNN = open('RNN/x_pad_sequences.pickle', 'rb')
feature_file_from_rnn = pickle.load(file_RNN)
file_RNN.close()

model_file_from_rnn = load_model('RNN/model.h5')

#load file sequences LSTM
file_LSTM = open('LSTM/x_pad_sequences.pickle', 'rb')
feature_file_from_lstm = pickle.load(file_LSTM)
file_LSTM.close()

model_file_from_lstm = load_model('LSTM/model.h5')

#endpoint CNNtext
@swag_from("docs/CNNtext.yml", methods=['POST'])
@app.route('/CNNtext', methods=['POST'])
def CNNtext():
    
    original_text = request.form.get('text')

    text = [cleansing(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    seq = pad_sequences(feature, maxlen=feature_file_from_cnn.shape[1])

    prediction = model_file_from_cnn.predict(seq)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis Using CNN",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

#endpoint CNNfile
@swag_from("docs/CNNfile.yml", methods=['POST'])
@app.route('/CNNfile', methods=['POST'])
def CNNfile():
    file = request.files["upload_file"]
    df = (pd.read_csv(file, encoding="cp1252"))
    df = df.rename(columns={df.columns[0]: 'text'})
    df['text_clean'] = df.apply(lambda row: cleansing(row['text']), axis=1)

    result = []

    for index, row in df.iterrows():
        text = tokenizer.texts_to_sequences([(row['text_clean'])])
        seq = pad_sequences(text, maxlen=feature_file_from_cnn.shape[1])
        prediction = model_file_from_cnn.predict(seq)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        result.append(get_sentiment)
    
    original = df.text_clean.to_list()

    json_response = {
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using CNN",
        'data' : {
            'text' : original,
            'sentiment' : result
        },
    }
    response_data = jsonify(json_response)
    return response_data

#endpoint RNNtext
@swag_from("docs/RNNtext.yml", methods=['POST'])
@app.route('/RNNtext', methods=['POST'])
def RNNtext():
    
    original_text = request.form.get('text')

    text = [cleansing(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    seq = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])

    prediction = model_file_from_rnn.predict(seq)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis Using RNN",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

#endpoint RNNfile
@swag_from("docs/RNNfile.yml", methods=['POST'])
@app.route('/RNNfile', methods=['POST'])
def RNNfile():
    file = request.files["upload_file"]
    df = (pd.read_csv(file, encoding="cp1252"))
    df = df.rename(columns={df.columns[0]: 'text'})
    df['text_clean'] = df.apply(lambda row: cleansing(row['text']), axis=1)

    result = []

    for index, row in df.iterrows():
        text = tokenizer.texts_to_sequences([(row['text_clean'])])
        seq = pad_sequences(text, maxlen=feature_file_from_rnn.shape[1])
        prediction = model_file_from_rnn.predict(seq)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        result.append(get_sentiment)
    
    original = df.text_clean.to_list()

    json_response = {
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using RNN",
        'data' : {
            'text' : original,
            'sentiment' : result
        },
    }
    response_data = jsonify(json_response)
    return response_data

#endpoint LSTMtext
@swag_from("docs/LSTMtext.yml", methods=['POST'])
@app.route('/LSTMtext', methods=['POST'])
def LSTMtext():
    
    original_text = request.form.get('text')

    text = [cleansing(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    seq = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])

    prediction = model_file_from_lstm.predict(seq)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis Using LSTM",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

#endpoint LSTMfile
@swag_from("docs/LSTMfile.yml", methods=['POST'])
@app.route('/LSTMfile', methods=['POST'])
def LSTMfile():
    file = request.files["upload_file"]
    df = (pd.read_csv(file, encoding="cp1252"))
    df = df.rename(columns={df.columns[0]: 'text'})
    df['text_clean'] = df.apply(lambda row: cleansing(row['text']), axis=1)

    result = []

    for index, row in df.iterrows():
        text = tokenizer.texts_to_sequences([(row['text_clean'])])
        seq = pad_sequences(text, maxlen=feature_file_from_lstm.shape[1])
        prediction = model_file_from_lstm.predict(seq)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        result.append(get_sentiment)
    
    original = df.text_clean.to_list()

    json_response = {
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using LSTM",
        'data' : {
            'text' : original,
            'sentiment' : result
        },
    }
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
	app.run(debug=True)