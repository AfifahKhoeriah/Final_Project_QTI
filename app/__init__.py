# Initial file for flask

# Called Flask library
from flask import Flask

# Used module name, used for another file can identified app folder
app = Flask(__name__) 

# Called routes file
# from app import main

app.static_folder = 'static'
