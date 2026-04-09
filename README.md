# E-commerce-Recommender-System
import pandas as pd

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Results
# E-commerce Reco...

#Sample data for user interactions and product features

#Product data

products=pd.DataFrame({

'product_id': [1, 2, 3, 4, 5],

'product_name': ['Laptop', 'Smartphone', 'Headphones', 'Camera', 'Smartwatch'],

'category': ['Electronics', 'Electronics', 'Accessories', 'Electronics', 'Wearable'],

'description': [

'High-performance laptop with great battery life.',

'Latest smartphone with high-resolution camera.',

'Noise-cancelling headphones for immersive sound.', 'Compact digital camera with excellent picture quality.',

'Smartwatch with health tracking features.'

]

})

# User purchase history (user-product interactions)

user_purchase_history= pd.DataFrame({

'user_id': [1, 1, 2, 2, 3, 3, 3, 4, 5],
'product_id': [1, 2, 2, 3, 3, 4, 5, 1, 2]

})

# Step 1: Content-Based Filtering using Product Descriptions

#Combine category and description for text-based similarity
products['text'] = products['category'] + products['description']
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(products['text'])

#calculate cosine similarity between products

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#Content-based recommendation function

def content_based_recommendations(product_id, cosine_sim=cosine_sim, products=products):

  idx = products[products['product_id'] == product_id].index[0]

  sim_scores = list(enumerate(cosine_sim[idx]))

  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

  sim_scores = sim_scores[1:4] # Get top 3 similar products

  product_indices = [i[0] for i in sim_scores]

  return products['product_name'].iloc[product_indices]

#Step 2: Collaborative Filtering using User Purchase History

#Create a user-item matrix

user_item_matrix = user_purchase_history.pivot_table(index='user_id', columns='product_id', aggfunc='size', fill_value=0)

#Compute user similarity

user_similarity = cosine_similarity(user_item_matrix)

user_sim_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

#Collaborative filtering recommendation function

def collaborative_recommendations(user_id, user_sim_df=user_sim_df, user_item_matrix= user_item_matrix):



  similar_users = user_sim_df[user_id].sort_values(ascending=False).index[1:4] # Top 3 similar users
  similar_users_data = user_item_matrix.loc[similar_users]
  user_purchases = user_item_matrix.loc[user_id]

  recommendations = similar_users_data.loc[:, user_purchases == 0].sum().sort_values(ascending=False)
  recommended_product_ids = recommendations.index[:3]

  return products[products['product_id'].isin(recommended_product_ids)]['product_name']

# Step 3: Recommendation Example

print("Content-Based Recommendations for Product 1:")

print(content_based_recommendations(1))

print("\nCollaborative Recommendations for User 1:")

print(collaborative_recommendations(1))
