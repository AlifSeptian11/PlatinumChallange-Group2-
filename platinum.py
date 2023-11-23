#import library
import re
import pandas as pd

from flask import Flask, jsonify, request

from flask import request
import flask
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