import pandas as pd
import numpy as np
import torch
from helper import CollabFNet, train_epocs

num_users = 100
num_items = 10


ratings_data = pd.read_csv('ratings.csv')

# Initialize and train the model with all data
model = CollabFNet(num_users, num_items, emb_size=50, n_hidden=20)
train_epocs(model, ratings_data, epochs=40, lr=0.01, wd=1e-6, unsqueeze=True)


# Save the trained model to a file
torch.save(model.state_dict(), 'recommendation_model.pth')



