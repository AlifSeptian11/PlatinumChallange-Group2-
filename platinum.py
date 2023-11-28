#import library
import pandas as pd
import re
import sqlite3 as sq
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import nltk.stem as stemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask, jsonify, request
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
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

def lowercase(text):
    return text.lower()

def remove_punctuation(text):
    text = re.sub(r'(?:\@|http?\://|https?\://|www)\S+', '', text) #menghapus https dan http
    text = re.sub('<.*?>', ' ', text) #mengganti karakter html dengan tanda petik
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) #menghilangkan semua karakter yang bukan huruf atau angka dan menggantinya dengan spasi.
    text = re.sub('\n',' ',text) #mengganti line baru dengan spasi
    text = re.sub(r':', ' ', text) #menggantikan karakter : dengan spasi 
    text = re.sub('gue','saya', text) # Mengganti kata "gue" dengan kata "saya"
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text) #menghapus single char
    text = ' '.join(text.split()) #memisahkan dan menggabungkan kata
    text = text.strip() #menghilangkan whitespace di awal dan di akhir teks
    text = re.sub(r'pic.twitter.com.[\w]+', '', text) #menghapus link picture
    text = re.sub(r'\buser\b',' ', text) #menghapus kata 'user'
    text = re.sub(r'\brt\b',' ', text) #menghapus awalan rt
    text = re.sub('RT',' ', text) #menghapus RT simbol
    text = re.sub(r'‚Ä¶', '', text)
    return text

## menghilangkan kata-kata yang tidak penting
#stopword_list = stopwords.words('indonesian')
#stopword_list.extend(['yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo',
#                       'kalo', 'amp', 'biar', 'bikin', 'bilang',
#                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
#                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
#                       'jd', 'jgn', 'sdh', 'aja', 'n', 't',
#                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
#                       '&amp', 'yah', 'dkk', 'xf', 'nku', 'url',
#                       'xa', 'xaa' 'xi', 'xe'])
#
## convert list ke dictionary
#stopword_list = set(stopword_list)
#
## remove stopwords pada list token
#def stopwords_removal(text):
#    text = [word for word in text if word not in stopword_list]
#    return text

# database
conn = sq.connect('C:/Users/Reza Fakhrurrozi/Documents/GitHub/PlatinumChallange-Group2-/database_pl.db', check_same_thread = False)
df_kamusalay = pd.read_sql_query('SELECT * FROM kamusalay', conn)

kamusalay = dict(zip(df_kamusalay['alay'], df_kamusalay['normal']))
def alay_to_normal(text):
    for word in df_kamusalay:
        return ' '.join([kamusalay[word] if word in kamusalay else word for word in text.split(' ')])

def text_cleansing(text):
    text = remove_punctuation(text)
    text = alay_to_normal(text)
    text = lowercase(text)
#    text = stopwords_removal(text)
    return text

@swag_from("docs/LSTMtext.yml", methods=['POST'])
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

@swag_from("docs/LSTMupload.yml", methods=['POST'])
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

if __name__ == '__main__':
	app.run(debug=True)