#import library
import re
import pandas as pd

from flask import Flask, jsonify, request

from flask import request
import flask
import sqlite3
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

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

conn = sqlite3.connect('C:/Users/Reza Fakhrurrozi/Binar Data Science/Gold Level Challenge/database_gold.db', check_same_thread = False)
df_kamusalay = pd.read_sql_query('SELECT * FROM kamusalay', conn)

def clean_text(text):
    text = re.sub('@[^\text]+', ' ', text) #menghapus username twitter
    text = re.sub(r'(?:\@|http?\://|https?\://|www)\S+', '', text) #menghapus https dan http
    text = re.sub('<.*?>', ' ', text) #mengganti karakter html dengan tanda petik
    text = re.sub('[^a-zA-Z]', ' ', text) #mempertimbangkan huruf dan nama
    text = re.sub('\n',' ',text) #mengganti line baru dengan spasi
    text = text.lower() #mengubah ke huruf kecil
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text) #menghapus single char
    text = ' '.join(text.split()) #memisahkan dan menggabungkan kata
    text = re.sub(r'pic.twitter.com.[\w]+', '', text)  #menghapus link picture
    text = re.sub('user',' ', text) #menghapus username
    text = re.sub('RT',' ', text) #menghapus RT simbol
    return text

kamusalay = dict(zip(df_kamusalay['alay'], df_kamusalay['normal']))
def alay_to_normal(text):
    for word in kamusalay:
        return ' '.join([kamusalay[word] if word in kamusalay else word for word in text.split(' ')])
    
def text_cleansing(text):
    text = clean_text(text)
    text = alay_to_normal(text)
    return text

@swag_from("docs/swagger_input.yml", methods=['POST'])
@app.route('/input_text', methods=['POST'])
def text_processing():
    input_txt = str(request.form["input_teks"])
    output_txt = text_cleansing(input_txt)

    conn.execute('create table if not exists input_teks (input_text varchar(255), output_text varchar(255))')
    query_txt = 'INSERT INTO input_teks (input_text, output_text) values (?,?)'
    val = (input_txt, output_txt)
    conn.execute(query_txt, val)
    conn.commit()

    return_txt = {"input":input_txt, "output": output_txt}
    return jsonify (return_txt)

@swag_from("docs/swagger_upload.yml", methods=['POST'])
@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files["upload_file"]
    df_csv = (pd.read_csv(file, encoding="cp1252"))

    df_csv['new_tweet'] = df_csv['Tweet'].apply(text_cleansing)
    df_csv.to_sql("clean_tweet", con=conn, index=False, if_exists='append')
    #conn.close()

    cleansing_tweet = df_csv.new_tweet.to_list()

    return_file = {
        'output': cleansing_tweet}
    return jsonify(return_file)