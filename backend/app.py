from collections import defaultdict
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
# print(item_list)
# print(Item.query.with_entities(Item.name).all()[0])


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

@app.route('/search', methods=['POST'])
def search_result():
    data = request.get_json()
    item_name = data[0]['item_name'].title()
    item = Item.query.filter_by(name=item_name).first()
    avg =  average_rating_window(item.id,32)
    result = [avg, item.positive_feedback, item.negative_feedback]
    return jsonify(result)

@app.route('/')
def index():
    list_of_items = [item.name.lower() for item in Item.query.all()]
    # ItemPage(1)
    top_items_this_week()
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
        list_of_items = [item.name for item in Item.query.all()]
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
    item = Item.query.get(item_id)
    item.average_rating = average_rating
    db.session.commit()
    return average_rating


def average_rating_window(item_id, window_size):
    # Query the database to get the average rating for the specified item
    ratings = (
        db.session.query(FoodReview.rating, FoodReview.timestamp)
        .filter(FoodReview.item_id == item_id)
        .filter(FoodReview.timestamp > datetime.utcnow()-timedelta(hours=5, minutes=30)-timedelta(days=window_size)).all()
    )
    ratings = [[rating.timestamp.date(), rating.rating] for rating in ratings]
    
    # sort ratings by date
    ratings = sorted(ratings, key=lambda x: x[0])
    
    # create this into a list of average ratings per day for the past 30 days
    rating_list = np.zeros(window_size+1)
    j = 0
    for i in range(window_size,0,-1):
        date_required = (datetime.utcnow()-timedelta(hours=5, minutes=30)-timedelta(days=i)).date()
        count=0
        while j<len(ratings) and ratings[j][0] == date_required:
            rating_list[i] += ratings[j][1]
            j += 1
            count += 1
        if count!=0:
            rating_list[i] /= count
    
    rating_list = rating_list[::-1]
    rating_list = rating_list[:-1]
    
    # generate a moving average of the ratings for 15 days from the current day
    # print(rating_list)
    final_list = []
    k = 0
    sum = 0
    count = 0
    for i in range(0,window_size//2):
        if rating_list[k] != 0:
            sum += rating_list[k]
            count += 1
        k += 1
    j = 0
    while k<len(rating_list):
        if rating_list[k] != 0:
            sum += rating_list[k]
            count += 1
        k += 1
        final_list.append(sum/count)
        if rating_list[j] != 0:
            sum -= rating_list[j]
            count -= 1
        j += 1
    # print(final_list, len(final_list),j,k)
    return final_list


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


def top_items_this_week():
    items = []
    # write a query to make a list of items with the highest average rating in the past week
    all_items = Item.query.all()
    reviews = FoodReview.query.filter(FoodReview.timestamp > datetime.utcnow()-timedelta(hours=5, minutes=30)-timedelta(days=7)).all()
    items_avg = np.zeros(len(all_items))
    count_list = np.zeros(len(all_items))
    for review in reviews:
        items_avg[review.item_id-1] += int(review.rating)
        count_list[review.item_id-1] += 1
    for i in range(len(items_avg)):
        if count_list[i] != 0:
            items_avg[i] /= count_list[i]
    # items_list = items_list.tolist()
    best5 = np.argsort(items_avg)[::-1][:5]
    for item in all_items:
        if item.id in best5:
            items.append({item.name: items_avg[item.id]})
    # sort items by average rating
    items = sorted(items, key=lambda x: list(x.values())[0], reverse=True)
    # print(items)
    return items


def top_reviews(item_id,topNum):
    # Query the database to get the top 5 reviews for the specified item
    top_reviews = (
        db.session.query(FoodReview)
        .filter(FoodReview.item_id == item_id)
        .order_by(FoodReview.sentiment_insights.desc())
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


def ItemPage(item_id):

    """Calculates the daily average rating and number of ratings for an item_id along wiht top reviews, lates reviews, and rating average.

    Args:
        item_id (int): The item_id to calculate the daily average rating and
            number of ratings for.

    Returns:
        list[list[float]]: A 2D array with the daily average rating and number
            of ratings for the item_id. Each sub-array contains the following
            elements:

                [0]: The date.
                [1]: The average rating.
                [2]: The number of ratings.
    """
    topReviews=top_reviews(item_id,5)
    latestReviews=latest_reviews(item_id,5) 
    avg_rating=get_average_rating(item_id)
    # reviews = FoodReview.query.filter_by(item_id=item_id).all()
    # grouped_reviews = defaultdict(list)
    # for review in reviews:
    #     grouped_reviews[review.timestamp.date()].append(review)
    # print(grouped_reviews)
    # l = average_rating_window(item_id, 32)
    # print(len(l),l )
    daily_average_ratings = average_rating_window(item_id, 32)
    # for date, reviews in grouped_reviews.items():
    #     average_rating = sum(review.rating for review in reviews) / len(reviews)
    #     number_of_ratings = len(reviews)
    #     daily_average_ratings[date]=average_rating
    # daily_average_ratings = sort(daily_average_ratings)
    final_data=[daily_average_ratings,topReviews ,latestReviews ,avg_rating]
    # print(final_data)
    return jsonify(data=final_data)

def ml_model():
    pass


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)