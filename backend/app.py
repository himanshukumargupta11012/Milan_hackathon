from datetime import timedelta
from functools import wraps
from flask import Flask, abort, render_template, redirect, url_for, request, jsonify, session
from dotenv import load_dotenv
import os
from db_models import *
from flask_login import login_user, LoginManager, current_user, logout_user
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from authlib.integrations.flask_client import OAuth
from authlib.common.security import generate_token
import numpy as np
import torch
import pandas as pd
from model import *
from scipy.sparse.linalg import svds
from pyabsa import AspectSentimentTripletExtraction as ASTE
from helper import CollabFNet, create_candidate_set, predict_ratings_for_candidate_set, recommend_items_for_user


from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
from nltk.tokenize import sent_tokenize
import nltk


def create_keyword_dict(item_id, df, n_keywords =20):
    # Filter the DataFrame to get reviews for the specified itemId
    item_reviews = df[df['itemId'] == item_id]['Review'].tolist()

    # Combine the reviews into a single text
    combined_text = ' '.join(item_reviews)

    # Tokenize the combined text into sentences
    sentences = sent_tokenize(combined_text)

    # Extract keywords from the combined text
    extracted_keywords = keywords(combined_text, words=n_keywords, lemmatize=True).split('\n')

    # Create a dictionary to store keyword occurrences
    keyword_occurrences = {keyword: [] for keyword in extracted_keywords}

    # Iterate through sentences and keywords
    for sentence in sentences:
        for keyword in extracted_keywords:
            if keyword in sentence:
                # Add the sentence to the dictionary under the keyword
                keyword_occurrences[keyword].append(sentence)

    return extracted_keywords, keyword_occurrences




def get_summary(item_id, df, proportion=0.2, max_words = 100):
    # Filter the DataFrame to get reviews for the specified itemId
    item_reviews = df[df['itemId'] == item_id]['Review'].tolist()

    # Combine the reviews into a single text
    combined_text = ' '.join(item_reviews)

    # Generate the summary
    summary_word_count = summarize(combined_text, word_count=max_words)
    summary_proportion = summarize(combined_text, ratio=proportion)
    summary = min(summary_word_count, summary_proportion, key=len)

    # Tokenize the summary into sentences
    summary_sentences = sent_tokenize(summary)

    # Remove duplicates by converting to a set and back to a list
    unique_summary_sentences = list(set(summary_sentences))

    # Reconstruct the summary with unique sentences
    unique_summary = ' '.join(unique_summary_sentences)

    return unique_summary













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
# Create a new engine instance
engine = create_engine('sqlite:///instance/canteen.db')
Session2 = sessionmaker(bind=engine)
session2 = Session2()
app.app_context().push()
db.create_all()


# aspect based sentiment analysis model
triplet_extractor = ASTE.AspectSentimentTripletExtractor(checkpoint="english")

# recommendation_model
num_users = User.query.count()
num_items = Item.query.count()
top_n = 5
item_list = [item.name for item in Item.query.all()]
reviews = [[item.review,item.item_id] for item in FoodReview.query.all()]

ratings = FoodReview.query.all()
# Create a DataFrame with the desired fields
ratings_data = pd.DataFrame([(rating.user_id, rating.item_id, rating.rating, rating.review) for rating in ratings],columns=['userId', 'itemId', 'rating', 'Review'])
ratings_data = ratings_data.groupby(['userId', 'itemId'])['rating'].mean().reset_index()
unique_items = ratings_data['itemId'].unique()
item_name_mapping = {item_id: item for item_id, item in enumerate(item_list, start=1)}



# Initialize and train the model with all data
model = CollabFNet(num_users, num_items, emb_size=50, n_hidden=20)
# train_epocs(model, ratings_data, epochs=40, lr=0.01, wd=1e-6, unsqueeze=True)

# Save the trained model to a file
torch.save(model.state_dict(), 'recommendation_model.pth')
# Load the saved model
loaded_model = CollabFNet(num_users, num_items, emb_size=50, n_hidden=20)
loaded_model.load_state_dict(torch.load('recommendation_model.pth'))

# Set the model to evaluation mode
loaded_model.eval()
candidate_set = create_candidate_set(ratings_data, num_users, num_items, item_name_mapping)
predicted_ratings_df = predict_ratings_for_candidate_set(loaded_model, candidate_set, item_name_mapping)
user_id_to_recommend = 3
user_recommendations = recommend_items_for_user(loaded_model, user_id_to_recommend, predicted_ratings_df, top_n)
recommend_item = user_recommendations.iloc[:, 1].values
print(recommend_item)


def require_login(f):
    @wraps(f)
    def innerfunction(*args, **kwargs):
        if current_user.is_authenticated:
            return f(*args, **kwargs)
        else:
            return redirect(url_for('login'))
    return innerfunction


def is_admin(f):
    @wraps(f)
    @require_login
    def innerfunction(*args, **kwargs):
        # print(current_user.email)
        if current_user.type==1 or current_user.type==2:
            return f(*args, **kwargs)
        else:
            return redirect(url_for('index',q=1))
    return innerfunction


def super_user(f):
    @wraps(f)
    @require_login
    def innerfunction(*args, **kwargs):
        if current_user.type==2: 
            return f(*args, **kwargs)
        else:
            return redirect(url_for('index',q=2))
    return innerfunction


def update_neg_pos(review, item_id): 
    dict = triplet_extractor.predict(review)['Triplets']

    if dict == '[]':
        return
    d2_list = np.array([[d['Aspect'], d['Opinion'], d['Polarity']] for d in dict], dtype="object")
    negative = []
    positive = []
    index = 0
    prev = ""
    for j in range(len(d2_list)):
        if np.all(d2_list[:, 0] == d2_list[0][0]) or np.all(d2_list[:, 1] == d2_list[0][1]):
            d2_list[j][1] = d2_list[j][1].replace(',', '').replace('.', '')
            if d2_list[j][2] == "Negative":
                if d2_list[j][1] not in negative:
                    negative.append(' '.join([d2_list[j][0], d2_list[j][1]]))

            if d2_list[j][2] == "Positive":
                if d2_list[j][1] not in positive:
                    positive.append(' '.join([d2_list[j][0], d2_list[j][1]]))
        
        else:
            if prev != d2_list[j][0]:
                d2_list[index][1] = d2_list[index][1].replace(',', '').replace('.', '')
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
    elif len(negative) > 0:
        item.negative_feedback += ',' + ','.join(negative)

    if item.positive_feedback == None:
        item.positive_feedback = ','.join(positive)
    elif len(positive) > 0:
        item.positive_feedback += ',' + ','.join(positive)

    db.session.commit()


# reviews = [[item.review,item.item_id] for item in FoodReview.query.all()]
# for i in reviews:
#     update_neg_pos(i[0], i[1])




@login_manager.user_loader
def load_user(id):
    return User.query.filter_by(id=id).first()


@app.route('/search', methods=['POST'])
def search_result():
    data = request.get_json()
    item_name = data[0]['item_name'].capitalize()
    item = Item.query.filter_by(name=item_name).first()

    

    avg =  average_rating_window(item.id,32)
    result = [avg, item.positive_feedback, item.negative_feedback]
    item_id = item.id

    keyword_list, keyword_dict = create_keyword_dict(item_id, ratings_data)
    item_summary = get_summary(item_id, ratings_data)

    stmt = session2.query(func.count(FoodReview.rating).label('total_quantity')).group_by(FoodReview.rating).filter(FoodReview.item_id == item_id).all()
    stmt = [i[0] for i  in stmt]
    result.append(stmt)
    result.append(keyword_dict)
    result.append(item_summary)
    return jsonify(result)


@app.route('/')
def index():
    list_of_items = [item.name.lower() for item in Item.query.all()]

    

    q_dict = {0:None,1:"You are not an admin", 2:"You are not a super user"}
    try :
        q = int(request.args.get('q'))
    except:
        q=0
    if not current_user.is_authenticated:
        return render_template('index.html', item_list=list_of_items, user=current_user, top5=top_items_this_week(), recommend_items=top_items_this_week(), messages=q_dict[q])
    else:
        item_id = recommend_user_items(current_user.id, top_n)
        print(item_id)
        item_name = [item_list[i-1] for i in item_id]
        return render_template('index.html', item_list=list_of_items, user=current_user, top5=top_items_this_week(), recommend_items=item_name, messages=q_dict[q])


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
        usr = User(email=user['email'], name=user['name'], profile_url=user['picture'],type=0)
        db.session.add(usr)
        db.session.commit()
    if usr.profile_url is None:
        usr.profile_url = user['picture']
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
        print(item_name)
        item_id=Item.query.filter_by(name=item_name).first().id
        review = request.form['review']
        update_neg_pos(review, item_id)
        rating = get_rating(review)
        newReview = FoodReview(user_id=user_id, review=review, rating=rating, item_id=item_id, sentiment_insights=None)
        db.session.add(newReview)
        db.session.commit()
        return redirect(url_for('index'))


@app.route('/admin', methods=['GET', 'POST'])
@is_admin
def admin():
    list_of_items = [item.name.lower() for item in Item.query.all()]
    if current_user.type == 2:
        return redirect('/super')
    return render_template('admin.html', user=current_user, top5=top_items_this_week(), item_list=list_of_items)


@app.route('/super', methods=['GET', 'POST'])
@super_user
def super():
    list_of_items = [item.name.lower() for item in Item.query.all()]
    return render_template('admin.html', user=current_user, top5=top_items_this_week(), item_list=list_of_items)


def average_rating_window(item_id, window_size):
    # Query the database to get the average rating for the specified item
    ratings = (
        db.session.query(FoodReview.rating, FoodReview.timestamp)
        .filter(FoodReview.item_id == item_id)
        .filter(FoodReview.timestamp > datetime.utcnow()+timedelta(hours=5, minutes=30)-timedelta(days=window_size)).all()
    )
    ratings = [[rating.timestamp.date(), rating.rating] for rating in ratings]
    
    # sort ratings by date
    ratings = sorted(ratings, key=lambda x: x[0])
    
    # create this into a list of average ratings per day for the past 30 days
    rating_list = np.zeros(window_size+1)
    j = 0
    for i in range(window_size,0,-1):
        date_required = (datetime.utcnow()+timedelta(hours=5, minutes=30)-timedelta(days=i)).date()
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
    return final_list


def top_items_this_week():
    items = {}
    # write a query to make a list of items with the highest average rating in the past week
    all_items = Item.query.all()
    reviews = FoodReview.query.filter(FoodReview.timestamp > datetime.utcnow()+timedelta(hours=5, minutes=30)-timedelta(days=7)).all()
    items_avg = np.zeros(len(all_items))
    count_list = np.zeros(len(all_items))
    for review in reviews:
        items_avg[review.item_id-1] += int(review.rating)
        count_list[review.item_id-1] += 1
    for i in range(len(items_avg)):
        if count_list[i] != 0:
            items_avg[i] /= count_list[i]
    best5 = np.argsort(items_avg)[::-1][:6]
    for item in all_items:
        if item.id in best5:
            items[item.name]= items_avg[item.id]
    # sort items by average rating
    items = dict(sorted(items.items(), key=lambda x: x[1], reverse=True))
    return items


# recommendation charan
def recommend_user_items(user_id,num_recommendations):
    ratings = FoodReview.query.all()
    df = pd.DataFrame([(rating.user_id, rating.item_id, rating.rating) for rating in ratings],columns=['user_id', 'item_id', 'rating'])
    df_avg = df.groupby(['user_id', 'item_id'])['rating'].mean().reset_index()
    
    # Create the user-item matrix
    user_item = pd.pivot_table(df_avg, index=['user_id'], columns='item_id', values='rating', fill_value=0)
    user_item.fillna(0,inplace = True)
    array_ind = user_item.index
    array_ind = list(array_ind)
    try:
        act_user_id = array_ind.index(user_id)
        # Singular Value Decomposition
        U, S, V = svds(user_item.values, k = 7)
        S = np.diag(S)
        predicted_ratings = np.matmul(np.matmul(U[act_user_id],S),V)
        
        # Sort the items by predicted rating.
        sorted_items = np.argsort(predicted_ratings)[::-1] + 1 # (item index starts from 1)
        
        # Recommend the top items.
        recommended_items = sorted_items[:num_recommendations]
        print(recommended_items)
        
        # Recommend new items
        old_items = np.nonzero(user_item.iloc[act_user_id].values)[0] + 1

        new_items  = recommended_items[~np.isin(recommended_items, old_items)]
        
        return new_items
    except:
        temp = top_items_this_week().keys()
        answer = []
        for i in temp:
            answer.append(item_list.index(i)+1)
        return answer
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
   