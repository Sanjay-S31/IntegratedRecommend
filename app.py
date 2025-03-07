import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

# Load Data from CSV Files
product_data = pd.read_csv('products.csv')
user_data = pd.read_csv('user_interactions.csv')
user_metadata = pd.read_csv('user_metadata.csv')

# 1. Collaborative Filtering (User-Item Matrix)
def collaborative_filtering_recommendations(user_id, top_n=3):
    # Create user-item matrix with ratings
    user_item_matrix = user_data.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
    
    # Check if user exists in the matrix
    if user_id not in user_item_matrix.index:
        print(f"User {user_id} not found in the dataset.")
        return []
    
    # Calculate user similarity using cosine similarity
    user_similarity = cosine_similarity(user_item_matrix)
    
    # Get the user's index in the matrix
    user_idx = list(user_item_matrix.index).index(user_id)
    
    # Get similar users (excluding the user itself)
    similar_users_indices = np.argsort(-user_similarity[user_idx])
    similar_users = [list(user_item_matrix.index)[i] for i in similar_users_indices if i != user_idx][:2]
    
    # Find products that similar users rated highly but the current user hasn't rated
    user_products = set(user_data[user_data['user_id'] == user_id]['product_id'])
    recommended_products = []
    
    for sim_user in similar_users:
        sim_user_products = set(user_data[user_data['user_id'] == sim_user]['product_id'])
        # Get products the similar user has but current user doesn't
        new_products = sim_user_products - user_products
        recommended_products.extend(list(new_products))
    
    # Get top N unique recommendations
    recommended_products = list(dict.fromkeys(recommended_products))[:top_n]
    
    print(f"Collaborative filtering for user {user_id}: {recommended_products}")
    return recommended_products

# 2. Content-Based Filtering (TF-IDF for Product Descriptions)
def content_based_recommendations(product_id, top_n=3):
    # Check if product exists
    if product_id not in product_data['product_id'].values:
        print(f"Product {product_id} not found in the dataset.")
        return []
    
    # Combine title and description for better content representation
    product_data['combined_text'] = product_data['title'] + " " + product_data['description'] + " " + product_data['category']
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(product_data['combined_text'])
    
    # Calculate content similarity
    idx = product_data[product_data['product_id'] == product_id].index[0]
    similarity_scores = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    
    # Get top N similar products (excluding the product itself)
    similar_indices = np.argsort(-similarity_scores)
    similar_indices = [i for i in similar_indices if i != idx][:top_n]
    
    similar_products = product_data.iloc[similar_indices]['product_id'].tolist()
    
    print(f"Content-based filtering for product {product_id}: {similar_products}")
    return similar_products

# 3. Category-Based Recommendations
def category_based_recommendations(product_id, top_n=3):
    # Check if product exists
    if product_id not in product_data['product_id'].values:
        print(f"Product {product_id} not found in the dataset.")
        return []
    
    # Get product category
    product_category = product_data[product_data['product_id'] == product_id]['category'].values[0]
    
    # Find products in the same category (excluding the product itself)
    category_products = product_data[
        (product_data['category'] == product_category) & 
        (product_data['product_id'] != product_id)
    ]['product_id'].tolist()
    
    # Sort by user rating
    category_products = sorted(
        category_products,
        key=lambda pid: product_data[product_data['product_id'] == pid]['user_rating'].values[0],
        reverse=True
    )[:top_n]
    
    print(f"Category-based filtering for product {product_id}: {category_products}")
    return category_products

# 4. Image-Based Recommendations
# Initialize ResNet model for image feature extraction
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_image_features(img_path):
    if not os.path.exists(img_path):
        return np.zeros((2048,))

    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return resnet_model.predict(img_array).flatten()
    except:
        return np.zeros((2048,))

# Precompute and normalize image features
image_features = np.array([extract_image_features(img) for img in product_data['image_path']])
scaler = MinMaxScaler()
image_features = scaler.fit_transform(image_features)

def image_based_recommendations(product_id, top_n=5):
    matching_indices = product_data.index[product_data['product_id'] == product_id].tolist()
    if not matching_indices:
        print(f"Product {product_id} not found in the dataset.")
        return []

    idx = matching_indices[0]
    similarity_scores = cosine_similarity([image_features[idx]], image_features).flatten()
    similar_indices = np.argsort(-similarity_scores)[1:top_n+1]
    
    similar_products = product_data.iloc[similar_indices]['product_id'].tolist()
    
    print(f"Image-based filtering for product {product_id}: {similar_products}")
    return similar_products

# 5. Context-Aware Recommendations
def context_aware_recommendations(user_id, product_id, top_n=3):
    # Check if user exists
    if user_id not in user_metadata['user_id'].values:
        print(f"User {user_id} not found in metadata.")
        return []
    
    # Get user purchase history
    purchase_history = user_metadata[user_metadata['user_id'] == user_id]['purchase_history'].values[0]
    purchase_history = [int(pid) for pid in purchase_history.split(',')]
    
    # Get recommendations from other methods
    collab_recs = collaborative_filtering_recommendations(user_id, top_n=5)
    content_recs = content_based_recommendations(product_id, top_n=5)
    category_recs = category_based_recommendations(product_id, top_n=5)
    image_recs = image_based_recommendations(product_id, top_n=5)
    
    # Combine recommendations
    all_recs = collab_recs + content_recs + category_recs + image_recs
    
    # Remove duplicates while preserving order
    seen = set()
    unique_recs = [x for x in all_recs if not (x in seen or seen.add(x))]
    
    # Sort by user rating
    final_recs = sorted(
        unique_recs,
        key=lambda pid: product_data[product_data['product_id'] == pid]['user_rating'].values[0],
        reverse=True
    )[:top_n]
    
    print(f"Context-aware recommendations for user {user_id} and product {product_id}: {final_recs}")
    return final_recs

# Unified Recommendation Engine
def unified_recommendation_engine(user_id, product_id, top_n=5):
    print(f"\nGenerating recommendations for User {user_id} based on Product {product_id}...")
    
    # Check if user and product exist
    user_exists = user_id in user_data['user_id'].values
    product_exists = product_id in product_data['product_id'].values
    
    if not user_exists:
        print(f"Warning: User ID {user_id} not found in the dataset.")
    if not product_exists:
        print(f"Warning: Product ID {product_id} not found in the dataset.")
    
    # If neither exists, return empty list
    if not user_exists and not product_exists:
        return []
    
    # Get recommendations from each method based on available data
    recommendations = []
    
    if user_exists:
        collab_recs = collaborative_filtering_recommendations(user_id, top_n=top_n)
        recommendations.extend(collab_recs)
    
    if product_exists:
        content_recs = content_based_recommendations(product_id, top_n=top_n)
        recommendations.extend(content_recs)
        
        category_recs = category_based_recommendations(product_id, top_n=top_n)
        recommendations.extend(category_recs)
        
        # Add image-based recommendations
        image_recs = image_based_recommendations(product_id, top_n=top_n)
        recommendations.extend(image_recs)
    
    if user_exists and product_exists:
        context_recs = context_aware_recommendations(user_id, product_id, top_n=top_n)
        recommendations.extend(context_recs)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_recs = [x for x in recommendations if not (x in seen or seen.add(x))]
    
    # Filter out the input product_id if present
    final_recs = [pid for pid in unique_recs if pid != product_id][:top_n]
    
    # Print detailed information for recommendations
    print("\nFinal Recommendations:")
    for idx, rec_id in enumerate(final_recs, 1):
        product_info = product_data[product_data['product_id'] == rec_id].iloc[0]
        print(f"{idx}. Product ID: {rec_id}, Title: {product_info['title']}, Category: {product_info['category']}, Rating: {product_info['user_rating']}")
    
    return final_recs

user_id = int(input("Enter User ID: "))
product_id = int(input("Enter Product ID: "))

recommended_products = unified_recommendation_engine(user_id, product_id, top_n=5)