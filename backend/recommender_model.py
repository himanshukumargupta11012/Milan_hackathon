import pandas as pd
import numpy as np
import torch
from helper import CollabFNet, train_epocs


# Speify the correct number of users and items
num_users = 130
num_items = 11


ratings = FoodReview.query.all()
# Create a DataFrame with the desired fields
ratings_data = pd.DataFrame([(rating.user_id, rating.item_id, rating.rating, rating.review) for rating in ratings],columns=['userId', 'itemId', 'rating', 'Review'])
ratings_data = ratings_data.groupby(['user_id', 'item_id'])['rating'].mean().reset_index()

# ratings_data = pd.read_csv('ratings.csv')

unique_items = ratings_data['Item'].unique()
item_id_mapping = {item: item_id for item_id, item in enumerate(unique_items, start=1)}
ratings_data['itemId'] = ratings_data['Item'].map(item_id_mapping)


# Initialize and train the model with all data
model = CollabFNet(num_users, num_items, emb_size=50, n_hidden=20)
train_epocs(model, ratings_data, epochs=40, lr=0.01, wd=1e-6, unsqueeze=True)


# Save the trained model to a file
torch.save(model.state_dict(), 'recommendation_model.pth')



