from flask import Flask, abort, render_template, redirect, url_for, request, jsonify
from dotenv import load_dotenv
import os, dotenv
from db_model import *
from flask_login import login_user, LoginManager, current_user, logout_user

load_dotenv()

app = Flask(__name__, template_folder='../frontend/html', static_folder='../frontend/static')
files = dotenv.load_dotenv(".env")
db_url = f"postgresql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)

# def authenticated(f):
#     def innerfunction(*args, **kwargs):
#         if current_user.is_authenticated:
#             return f(*args, **kwargs)
#         return abort(403)
#     return innerfunction

@login_manager.user_loader
def load_user(id):
    return Users.query.filter_by(id=id).first()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method=='GET':
        return render_template('login.html')
    else:
        username = request.form.get('username')
        password = request.form.get('password')
        user = Users.query.filter_by(username=username).first()
        if user is None or not user.check_password(password):
            return redirect(url_for('login'))
        login_user(user)
        return redirect(url_for('index'))
    
@app.route('/review')
def review():
    return render_template('review.html')


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