import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine

# Load product data (Assuming a dataset with columns: 'product_id', 'title', 'description', 'category', 'image_path', 'user_rating')
product_data = pd.read_csv("products.csv")

# Load user interaction data
user_data = pd.read_csv("user_interactions.csv")

# Load user metadata for context-aware recommendations
user_metadata = pd.read_csv("user_metadata.csv")

# 1. Collaborative Filtering (User-Item Matrix)
def collaborative_filtering_recommendations(user_id, top_n=5):
    user_item_matrix = user_data.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    
    if user_id not in user_item_matrix.index:
        return []

    similar_users = np.argsort(-user_similarity[user_item_matrix.index.get_loc(user_id)])[:top_n]
    recommended_products = set()
    
    for sim_user in similar_users:
        sim_user_products = user_item_matrix.iloc[sim_user].nlargest(top_n).index
        recommended_products.update(sim_user_products)
    
    return list(recommended_products)

# 2. Content-Based Filtering (TF-IDF for Product Descriptions)
vectorizer = TfidfVectorizer(stop_words='english')
product_data['combined_text'] = product_data['title'] + " " + product_data['description']
tfidf_matrix = vectorizer.fit_transform(product_data['combined_text'])

def content_based_recommendations(product_id, top_n=5):
    matching_indices = product_data.index[product_data['product_id'] == product_id].tolist()

    if not matching_indices:  # Check if the list is empty
        print(f"Error: Product ID {product_id} not found in the dataset.")
        return []  # Return an empty list instead of crashing

    idx = matching_indices[0]  # Get the first matching index
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_products = np.argsort(-cosine_sim)[1:top_n+1]
    
    return product_data.iloc[similar_products]['product_id'].tolist()


# 3. Image-Based Recommendations (Using ResNet50 for Feature Extraction)
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_image_features(img_path):
    if not os.path.exists(img_path):
        print(f"Error: Image path does not exist - {img_path}")
        return np.zeros((2048,))  # Return a zero vector of fixed size

    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = resnet_model.predict(img_array).flatten()
        
        # Ensure features have a fixed shape
        if features.shape[0] != 2048:
            print(f"Warning: Feature shape mismatch for {img_path}")
            return np.zeros((2048,))  # Return a zero vector of fixed size

        return features
    
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return np.zeros((2048,))  # Return a zero vector of fixed size

    

image_features = np.array([extract_image_features(img) for img in product_data['image_path']])

def image_based_recommendations(product_id, top_n=5):
    idx = product_data.index[product_data['product_id'] == product_id].tolist()[0]
    similarity_scores = cosine_similarity([image_features[idx]], image_features).flatten()
    similar_products = np.argsort(-similarity_scores)[1:top_n+1]
    return product_data.iloc[similar_products]['product_id'].tolist()

# 4. Context-Aware Recommendations (Hybrid Approach)
def context_aware_recommendations(user_id, top_n=5):
    user_metadata['user_id'] = user_metadata['user_id'].astype(int)  # Ensure IDs match type
    user_info = user_metadata[user_metadata['user_id'] == user_id].iloc[0]

    # Convert purchase history to a list
    purchase_history_list = user_info['purchase_history'].split(",")  # Ensure it's a list

    # Combine multiple recommendation scores
    collab_recs = collaborative_filtering_recommendations(user_id, top_n)
    if not collab_recs:
        return []  # Ensure we have at least one recommendation

    content_recs = content_based_recommendations(collab_recs[0], top_n) if collab_recs else []
    image_recs = image_based_recommendations(collab_recs[0], top_n) if collab_recs else []

    final_recs = list(set(collab_recs + content_recs + image_recs))

    # Sort based on user preferences
    final_recs = sorted(final_recs, key=lambda pid: purchase_history_list.count(str(pid)), reverse=True)

    return final_recs[:top_n]


# Unified Recommendation System Function
def unified_recommendation_engine(user_id, product_id, top_n=5):
    print(f"Generating recommendations for User {user_id} based on Product {product_id}...")
    
    collab_recs = collaborative_filtering_recommendations(user_id, top_n)
    content_recs = content_based_recommendations(product_id, top_n)
    image_recs = image_based_recommendations(product_id, top_n)
    context_recs = context_aware_recommendations(user_id, top_n)
    
    final_recommendations = list(set(collab_recs + content_recs + image_recs + context_recs))
    return final_recommendations[:top_n]

# Example Usage
user_id = 102
product_id = 10
recommended_products = unified_recommendation_engine(user_id, product_id, top_n=5)
print("Recommended Products:", recommended_products)
