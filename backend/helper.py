import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import itertools



def proc_col(col, train_col=None):
    """
    Encodes a pandas column with continuous IDs.

    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    
    # Create a dictionary that maps category names to numerical IDs
    name2idx = {o: i for i, o in enumerate(uniq)}
    
    # Replace each value in the column with its corresponding numerical ID,
    # using -1 if the value is not found in 'train_col'
    encoded_col = np.array([name2idx.get(x, -1) for x in col])
    
    # Calculate the number of unique categories in the column
    num_uniq = len(uniq)
    
    return name2idx, encoded_col, num_uniq



def encode_data(df, train=None):
    """
    Encodes rating data with continuous user and item IDs.

    """
    df = df.copy()
    for col_name in ["userId", "itemId"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _, col, _ = proc_col(df[col_name], train_col)
        
        # Remove rows with negative IDs (IDs not found in 'train' data)
        df = df[df[col_name] >= 0]
        
        # Update the DataFrame with the encoded column
        df[col_name] = col
    
    return df


def train_epocs(model, df_train, epochs=10, lr=0.01, wd=0.0, unsqueeze=False):
    """
    Training loop for a recommendation model.

    This function trains the recommendation model for the specified number of epochs.
    It uses Mean Squared Error (MSE) loss and an Adam optimizer for training.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    
    for i in range(epochs):
        # Load user IDs, item IDs, and ratings from the training data
        users = torch.LongTensor(df_train.userId.values - 1)
        items = torch.LongTensor(df_train.itemId.values - 1)
        ratings = torch.FloatTensor(df_train.rating.values)
        
        # Optionally, unsqueeze the ratings tensor to match the model output shape
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        
        # Forward pass: compute predicted ratings
        y_hat = model(users, items)
        
        # Calculate Mean Squared Error (MSE) loss between predictions and actual ratings
        loss = F.mse_loss(y_hat, ratings)
        
        # Backpropagation: compute gradients and update model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print the loss for the current epoch
        print(f"Epoch {i+1}/{epochs}, Loss: {loss.item()}")





class CollabFNet(nn.Module):
    def __init__(self, num_users, num_items, emb_size=2, n_hidden=3):   # emb_size and n_hidden should be hyperparameters
        """
        Collaborative Filtering Neural Network (CollabFNet) model for recommendation.

        Initializes the CollabFNet model with user and item embedding layers and neural network layers.
        """
        super(CollabFNet, self).__init__()
        
        self.user_emb = nn.Embedding(num_users, emb_size)
        
        self.item_emb = nn.Embedding(num_items, emb_size)
        
        self.lin1 = nn.Linear(emb_size * 2, n_hidden)
        
        self.lin2 = nn.Linear(n_hidden, 1)
        
        self.drop1 = nn.Dropout(0.1)
        
    def forward(self, u, v):
        """
        Forward pass of the CollabFNet model.

        Parameters:
        - u: torch.Tensor
          Tensor containing user IDs.
        - v: torch.Tensor
          Tensor containing item IDs.

        Returns:
        - torch.Tensor
          Predicted ratings or scores for user-item interactions.
        """
        # Lookup user embeddings for the given user IDs
        U = self.user_emb(u)
        
        # Lookup item embeddings for the given item IDs
        V = self.item_emb(v)
        
        # Concatenate user and item embeddings
        x = F.relu(torch.cat([U, V], dim=1))
        
        # Apply dropout for regularization
        x = self.drop1(x)
        
        # Apply ReLU activation to the first linear layer
        x = F.relu(self.lin1(x))
        
        # Apply the second linear layer for final predictions
        x = self.lin2(x)
        
        return x



def create_candidate_set(ratings_df, num_users, num_items, item_name_mapping):

    # Create a set of all possible user-item pairs
    all_user_ids = range(1, num_users + 1)          # 1- 100   # since ratings_df will consist of 1-100
    all_item_ids = range(1, num_items + 1)
    all_user_item_pairs = list(itertools.product(all_user_ids, all_item_ids))

    # Convert the rated user-item pairs to a set for faster lookup
    rated_user_item_pairs = set(zip(ratings_df['userId'], ratings_df['itemId']))

    # Identify unused user-item pairs as the complement of rated pairs
    unused_user_item_pairs = list(set(all_user_item_pairs) - rated_user_item_pairs)

    # Create a DataFrame for the candidate set
    candidate_set = pd.DataFrame(unused_user_item_pairs, columns=['userId', 'itemId'])
    candidate_set['item'] = candidate_set['itemId'].map(item_name_mapping)
    
    return candidate_set   # candidate set will have 1-100 and 1-10 combinations




# Function to predict ratings for the candidate set using the trained model
def predict_ratings_for_candidate_set(model, candidate_set, item_name_mapping):
    """
    Predict ratings for user-item pairs in the candidate set.
    """
    # Convert user and item IDs to PyTorch tensors
    user_ids = torch.LongTensor(candidate_set['userId'].values - 1)         # for prediction, we subtract 1   
    item_ids = torch.LongTensor(candidate_set['itemId'].values - 1)
    
    # Use the model to predict ratings
    predicted_ratings = model(user_ids, item_ids)
    
    results_df = pd.DataFrame({
        'userId': candidate_set['userId'],         # but for df we can have 1-100
        'itemId': candidate_set['itemId'], 
        'item': candidate_set['item'] ,
        'predicted_ratings': predicted_ratings.squeeze().tolist()
    })
    
    return results_df


# Function to recommend items for a specific user
def recommend_items_for_user(model, user_id, results_df, top_n=3):
    """
    Recommend items for a specific user based on predicted ratings.

    """
    # Filter candidate set for the specific user
    user_candidate_set = results_df[results_df['userId'] == user_id]
    
    # Sort the user's candidate set by predicted rating in descending order
    sorted_candidate_set = user_candidate_set.sort_values(by='predicted_ratings', ascending=False)
    
    # Select the top-N recommendations
    recommendations = sorted_candidate_set.head(top_n)
    
    return recommendations




