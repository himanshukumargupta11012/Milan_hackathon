from flask import Flask, abort, render_template, redirect, url_for, request, jsonify, session
from dotenv import load_dotenv
import os, requests
from db_models import *
from flask_login import login_user, LoginManager, current_user, logout_user
from authlib.integrations.flask_client import OAuth
from authlib.common.security import generate_token

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

item_list = [item.name for item in Item.query.all()]
print(item_list)
@login_manager.user_loader
def load_user(id):
    return User.query.filter_by(id=id).first()

list_of_items = ["tea", "coffee", "veg noodles", "non-veg noodles"]
@app.route('/')
def index():
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
    print(" Google User ", user)
    print(current_user.is_authenticated)
    return redirect('/')


@app.route('/logout')
def logout():
    logout_user()
    print("Logged out")
    return redirect(url_for('index'))

@app.route('/review', methods=['GET', 'POST'])
def get_review():
    if request.method=='GET':
        return render_template('review.html')
    else:
        feedback = request.form.get('feedback')
        # add to database
        return redirect(url_for('index'))

def ml_model():
    pass


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)