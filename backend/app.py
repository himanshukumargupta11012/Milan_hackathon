from flask import Flask, abort, render_template, redirect, url_for, request, jsonify, session
from dotenv import load_dotenv
import os, requests, json
from db_models import *
from flask_login import login_user, LoginManager, current_user, logout_user
from authlib.integrations.flask_client import OAuth
from authlib.common.security import generate_token

app = Flask(__name__, template_folder='../frontend/html', static_folder='../frontend/static')
load_dotenv(".env")
db_url = f"sqlite:///canteen.db"

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
oauth = OAuth(app)
CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']
# db.create_all(app.app_context())
# DATA = {
#         'response_type':"code", # this tells the auth server that we are invoking authorization workflow
#         'redirect_uri':"https://localhost:5000/google/auth", # redirect URI https://console.developers.google.com/apis/credentials
#         'scope': 'https://www.googleapis.com/auth/userinfo.email', # resource we are trying to access through Google API
#         'client_id':CLIENT_ID, # client ID from https://console.developers.google.com/apis/credentials
#         'prompt':'consent'} # adds a consent screen
 
# URL_DICT = {
#         'google_oauth' : 'https://accounts.google.com/o/oauth2/v2/auth', # Google OAuth URI
#         'token_gen' : 'https://oauth2.googleapis.com/token', # URI to generate token to access Google API
#         'get_user_info' : 'https://www.googleapis.com/oauth2/v3/userinfo' # URI to get the user info
#         }
 
# Create a Sign in URI
# CLIENT = oauth2.WebApplicationClient(CLIENT_ID)
# REQ_URI = CLIENT.prepare_request_uri(
#     uri=URL_DICT['google_oauth'],
#     redirect_uri=DATA['redirect_uri'],
#     scope=DATA['scope'],
    # prompt=DATA['prompt'])

# def authenticated(f):
#     def innerfunction(*args, **kwargs):
#         if current_user.is_authenticated:
#             return f(*args, **kwargs)
#         return abort(403)
#     return innerfunction

@login_manager.user_loader
def load_user(id):
    return User.query.filter_by(id=id).first()

@app.route('/')
def index():
    return render_template('index.html')

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
    print(redirect_uri)
    session['nonce'] = generate_token()
    return oauth.google.authorize_redirect(redirect_uri, nonce=session['nonce'])

@app.route('/google/auth')
def google_auth():
    token = oauth.google.authorize_access_token()
    user = oauth.google.parse_id_token(token, nonce=session['nonce'])
    session['user'] = user
    usr = User.query.filter_by(email=user['email']).first()
    login_user(usr, remember=True)
    print(" Google User ", user)
    print(current_user.is_authenticated)
    return redirect('/')


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/feedback', methods=['GET', 'POST'])
def get_feedback():
    if request.method=='GET':
        return render_template('feedback.html')
    else:
        feedback = request.form.get('feedback')
        # add to database
        return redirect(url_for('index'))

def ml_model():
    pass


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)