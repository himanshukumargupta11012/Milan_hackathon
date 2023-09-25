from flask import flask
from flask_sqlalchemy import flask_sqlalchemy
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

db_url = f"postgresql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"

app.config['SQLALCHEMY_DATABASE_URI'] = db_url

db = SQLAlchemy(app)

# class User(db.Model):
#     id = db