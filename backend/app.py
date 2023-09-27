from datetime import timedelta
from functools import wraps
from flask import Flask, abort, render_template, redirect, url_for, request, jsonify, session
from dotenv import load_dotenv
import os, requests
from db_models import *
from flask_login import login_user, LoginManager, current_user, logout_user
from authlib.integrations.flask_client import OAuth
from authlib.common.security import generate_token
import numpy as np

# from pyabsa import AspectSentimentTripletExtraction as ASTE
# import numpy as  np

# triplet_extractor = ASTE.AspectSentimentTripletExtractor(
#     checkpoint="english"
# )

# print(triplet_extractor.predict("maggi is good"))

load_dotenv(".env")
CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']

app = Flask(__name__, template_folder='../frontend/html', static_folder='../frontend/static')
db_url = "sqlite:///canteen.db"
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
oauth = OAuth(app)

app.app_context().push()
db.create_all()

def require_login(f):
    @wraps(f)
    def innerfunction(*args, **kwargs):
        if current_user.is_authenticated:
            return f(*args, **kwargs)
        else:
            return redirect(url_for('login'))
    return innerfunction


def RunModelSentimentAnalysis():
    return 4

# print(item_list)
item_list = [item.name for item in Item.query.all()]
print(item_list)
print(Item.query.with_entities(Item.name).all()[0])


reviews = [[item.review,item.item_id] for item in FoodReview.query.all()]


def update_neg_pos(review, item_id):

    dict = triplet_extractor.predict(review)['Triplets']
    print(len(dict))
    if dict == '[]':
        return
    d2_list = np.array([[d['Aspect'], d['Opinion'], d['Polarity']] for d in dict], dtype="object")
    negative = []
    positive = []
    index = 0
    prev = ""
    for j in range(len(d2_list)):
        if np.all(d2_list[:, 0] == d2_list[0][0]) or np.all(d2_list[:, 1] == d2_list[0][1]):
            if d2_list[j][2] == "Negative":
                if d2_list[j][1] not in negative:
                    negative.append(' '.join([d2_list[j][0], d2_list[j][1]]))

            if d2_list[j][2] == "Positive":
                if d2_list[j][1] not in positive:
                    positive.append(' '.join([d2_list[j][0], d2_list[j][1]]))
        
        else:
            if prev != d2_list[j][0]:
                if d2_list[index][2] == "Negative":
                    if d2_list[index][2] not in negative:
                        negative.append(' '.join([d2_list[j][0], d2_list[index][1]]))

                if d2_list[index][2] == "Positive":
                    if d2_list[index][1] not in positive:
                        positive.append(' '.join([d2_list[j][0], d2_list[index][1]]))
                index += 1
                prev = d2_list[j][0]
    
    item = Item.query.get(item_id)

    if item.negative_feedback == None:
        item.negative_feedback = ','.join(negative)
    else:
        item.negative_feedback += ',' + ','.join(negative)

    if item.positive_feedback == None:
        item.positive_feedback = ','.join(positive)
    else:
        item.positive_feedback += ',' + ','.join(positive)

    db.session.commit()

    print(negative, positive)

# for i in reviews:
#     update_neg_pos(i[0], i[1])




@login_manager.user_loader
def load_user(id):
    return User.query.filter_by(id=id).first()

@app.route('/')
def index():
    if request.method == 'POST': 
        user_id = request.form['user_id']
        item_name = request.form['item_name']
        item_id=Item.query.filter_by(name=item_name).first().id
        review = request.form['review']
        rating = int(request.form['rating'])
        sentiment_insights = RunModelSentimentAnalysis(review)
        try: 
            # Validate the data if needed
            newReview = FoodReview(user_id=user_id, review=review, rating=rating, item_id=item_id, sentiment_insights=sentiment_insights)
            db.session.add(newReview)
            db.session.commit()
            return render_template('index.html', data={'status': True})
        except Exception as e:
            print(e)
            return render_template('index.html', data={'status': False, 'error_message': str(e)})
    print(latest_reviews(2,3))
    return render_template('index.html', item_list=list_of_items, user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    CONF_URL = 'https://accounts.google.com/.well-known/openid-configuration'
    oauth.register(
        name='google',
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        server_metadata_url=CONF_URL,
        client_kwargs={
            'scope': 'openid email profile'
        }
    )
    redirect_uri = url_for('google_auth', _external=True)
    session['nonce'] = generate_token()
    return oauth.google.authorize_redirect(redirect_uri, nonce=session['nonce'])

@app.route('/google/auth')
def google_auth():
    token = oauth.google.authorize_access_token()
    user = oauth.google.parse_id_token(token, nonce=session['nonce'])
    session['user'] = user
    usr = User.query.filter_by(email=user['email']).first()
    if usr is None:
        usr = User(email=user['email'], name=user['name'], profile_url=user['picture'])
        db.session.add(usr)
        db.session.commit()
    login_user(usr, remember=True)
    return redirect('/')


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/review', methods=['GET', 'POST'])
@require_login
def get_review():
    if request.method=='GET':
        return render_template('review.html')
    elif request.method == 'POST': 
        user_id = current_user.id
        item_name = request.form['contentItem']
        item_id=Item.query.filter_by(name=item_name).first().id
        review = request.form['review']
        # rating = int(request.form['rating'])
        rating = np.random.randint(3, 5)
        # sentiment_insights = RunModelSentimentAnalysis(review)
        # timestamp = datetime.utcnow()+timedelta(hours=5, minutes=30)
        newReview = FoodReview(user_id=user_id, review=review, rating=rating, item_id=item_id, sentiment_insights=None)
        db.session.add(newReview)
        db.session.commit()
        return render_template('index.html', user=current_user, item_list=list_of_items)

def get_average_rating(item_id):
    # Query the database to get the average rating for the specified item
    average_rating = (
        db.session.query(db.func.avg(FoodReview.rating))
        .filter(FoodReview.item_id == item_id).scalar()
    )
    return average_rating


def latest_reviews(item_id,topNum):
    # Query the database to get the top 5 reviews for the specified item
    top_reviews = (
        db.session.query(FoodReview)
        .filter(FoodReview.item_id == item_id)
        .order_by(FoodReview.timestamp.asc())
        .limit(topNum)
        .all()
    )

    # Create a list of dictionaries to hold the review details
    top_reviews_list = []
    for review in top_reviews:
        top_reviews_list.append({
            'id': review.id,
            'item_id': review.item_id,
            'review': review.review,
            'rating': review.rating,
            'user_id': review.user_id,
            'sentiment_insights': review.sentiment_insights,
            'timestamp': review.timestamp.isoformat(),
        })

    return top_reviews_list

def ml_model():
    pass


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)