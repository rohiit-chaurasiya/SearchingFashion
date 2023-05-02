import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set the paths for the extracted clothing items and the folder to store the results
extracted_clothing_items_path = "clothing_items"
top_matches_folder_path = "static/top_matches"

# Load the reference database of clothing images
reference_database_path = "image"
reference_database = {}
for root, dirs, files in os.walk(reference_database_path):
    for file in files:
        if file.endswith(".jpg"):
            image_path = os.path.join(root, file)
            image = Image.open(image_path)
            # Resize the image to the same size as the clothing item
            image = image.resize((224, 224))
            image_array = np.array(image)
            # Reshape the image array to have the same number of columns as the clothing item array
            image_array = image_array.reshape(-1, 224 * 224 * 3)
            reference_database[image_path] = image_array

# Loop over the extracted clothing items
for clothing_item_file in os.listdir(extracted_clothing_items_path):
    
    clothing_item_path = os.path.join(extracted_clothing_items_path, clothing_item_file)
    clothing_item = Image.open(clothing_item_path)
    
    resized_clothing_item = clothing_item.resize((224, 224))
    
    clothing_item_array = np.array(resized_clothing_item)
    
    clothing_item_array = clothing_item_array.reshape(-1, 224 * 224 * 3)
    # Calculate the cosine similarity between the clothing item and each reference image
    similarity_scores = []
    for reference_image_path, reference_image_array in reference_database.items():
        # Calculate the cosine similarity and append the result to the list
        similarity_score = cosine_similarity(clothing_item_array, reference_image_array)
        similarity_scores.append((reference_image_path, similarity_score))
    # Sort the similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
   
    item_name = os.path.splitext(clothing_item_file)[0]
    item_folder_path = top_matches_folder_path
    if not os.path.exists(item_folder_path):
        os.makedirs(item_folder_path)
    # Save the top matching reference images for the current clothing item
    for i in range(1):
        if i < len(similarity_scores):
            top_match_path = os.path.join(item_folder_path, "{}_top_match_{}.jpg".format(item_name, i+1))
            top_match_image = Image.open(similarity_scores[i][0])
            top_match_image.save(top_match_path)
