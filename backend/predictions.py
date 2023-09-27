import pandas as pd
import numpy as np
import torch
from helper import CollabFNet, create_candidate_set, predict_ratings_for_candidate_set, recommend_items_for_user


num_users = 100
num_items = 10
top_n=3

ratings_data = pd.read_csv('ratings.csv')

unique_items = ratings_data['Item'].unique()
item_id_mapping = {item: item_id for item_id, item in enumerate(unique_items, start=1)}
ratings_data['itemId'] = ratings_data['Item'].map(item_id_mapping)
item_name_mapping = {item_id: item_name for item_name, item_id in item_id_mapping.items()}

# Load the saved model
loaded_model = CollabFNet(num_users, num_items, emb_size=50, n_hidden=20)
loaded_model.load_state_dict(torch.load('recommendation_model.pth'))

# Set the model to evaluation mode
loaded_model.eval()

candidate_set = create_candidate_set(ratings_data, num_users, num_items, item_name_mapping)
# print(candidate_set)
predicted_ratings_df = predict_ratings_for_candidate_set(loaded_model, candidate_set, item_name_mapping)

# Recommend items for a specific user (e.g., user with userId=1)
user_id_to_recommend = 3
user_recommendations = recommend_items_for_user(loaded_model, user_id_to_recommend, predicted_ratings_df, top_n)
