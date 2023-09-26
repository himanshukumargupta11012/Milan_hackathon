from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import DateTime

# app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://username:password@localhost/database_name'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:1234@localhost:5432/milan'
 
db = SQLAlchemy()


# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(80), nullable=True)
    
    # Define a one-to-many relationship with FoodReview
    reviews = db.relationship('FoodReview', backref='user', lazy=True)

# Food Review Model
class FoodReview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('item.id'), nullable=False)
    review = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sentiment_insights = db.Column(db.Float)  # Integer for sentiment insights
    timestamp = db.Column(DateTime, default=datetime.utcnow)

# Item Model
class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    average_rating = db.Column(db.Float) 
    item_image_url = db.Column(db.String(255)) 
    
    # Define a one-to-many relationship with FoodReview
    reviews = db.relationship('FoodReview', backref='item', lazy=True)

# Create the tables in the database

# db = SQLAlchemy()

# table 1 user
# 1. user email address(pk)
# 2. User NAME
# 3. user id
    
# table 2 food reviews and ratings for each user
# 1. Rewiew id(pk)
# 2. Item id(fk)
# 3. Item Name
# 4. Review
# 5. Rating
# 6.user id(fk)
# 7. Timestamp(not needed may be)
# 8. Sentiment insights(int variable)

# table eachitem
# 1. Item id(pk)
# 2. Item Name
# 3. AverageRating
# 4. Item Image

# table 4 item ratings and sentiment
# 1. Review id(pk,fk)
# 2. Item id(fk)
# 3. Sentiment insights()
# 4. Timestamp(fk)
#  #  to order according to latest
#  # 7. Timestamp(not needed may be)

