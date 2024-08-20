from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from faiss_service import FaissService  # Import your FAISS service
from chat_gpt_service import ChatGPTService  # Import your ChatGPT service

# Data to be indexed
data = [
    "Red sports car with leather seats",
    "Luxury SUV with sunroof and heated seats",
    "Electric car with autonomous driving features",
    "Compact sedan with high fuel efficiency",
    "Motorbike with high acceleration and sleek design"
]

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to encode the texts
def encode_text(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

# Encode the data
embeddings = encode_text(data)

# Initialize the FAISS service
dim = embeddings.shape[1]  # Dimension of the embeddings
faiss_service = FaissService(dim=dim)

# Add embeddings to the FAISS index
faiss_service.add_to_index(embeddings)

# Save the index and database
faiss_service.save_index()

# Function to perform search with a threshold
def search(query, threshold=None):
    query_embedding = encode_text([query])
    distances, indices = faiss_service.search_index(query_embedding, top_k=5)
    
    # Filter results based on threshold if provided
    if threshold is not None:
        results = [(data[idx], distance) for i, (idx, distance) in enumerate(zip(indices[0], distances[0])) if distance <= threshold]
    else:
        results = [(data[idx], distance) for i, (idx, distance) in enumerate(zip(indices[0], distances[0]))]
    
    return results

# Example search with threshold
user_query = "Which cars are autonomous?"
threshold = 30  # Set your threshold value here
search_results = search(user_query, threshold=threshold)

# Initialize the ChatGPT service
chatgpt_service = ChatGPTService()

# Generate a response based on the search results
response = chatgpt_service.generate_response(prompt=user_query, semantic_result=search_results)

# Display the results
print("Search results:")
for i, (result, distance) in enumerate(search_results):
    print(f"{i + 1}: {result} (Distance: {distance})")

print("\nGenerated Response:")
print(response)
