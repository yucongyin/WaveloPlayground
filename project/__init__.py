from flask import Flask
from flask_socketio import SocketIO

# Create a Flask web server from the flaskr Flask web server.
app = Flask(__name__)

# Create a SocketIO server from the Flask application.
socketio = SocketIO(app)
