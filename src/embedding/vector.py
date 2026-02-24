from sentence_transformers import SentenceTransformer

from lib.utils import  vector_magnitude

class Vector:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def add_vectors(self, vec1, vec2):
        
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")
        
        return [vec1[i] + vec2[i] for i in range(len(vec1))]

    def subtract_vectors(self, vec1, vec2):

        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")
        
        return [vec1[i] - vec2[i] for i in range(len(vec1))]
    
    def dot_product(self, vec1, vec2):

        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")

        sum: float = 0
        
        for i in range(len(vec1)):
            sum += vec1[i] * vec2[i] 
        
        return sum

    def cosine_similar(self, vec1, vec2):

        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")

        magnitude_vec1 = vector_magnitude(vec1)
        magnitude_vec2 = vector_magnitude(vec2)

        if magnitude_vec1 == 0 or magnitude_vec2 == 0:
            return 0.0
        
        dot_product = self.dot_product(vec1, vec2)
        
        return dot_product/(magnitude_vec1*magnitude_vec2)

    def generate_embedding(self, text):
        if not text or text.strip() == "":
            raise ValueError("Input text cannot be empty or contain only whitespace.")
        embedding = self.model.encode(text)
        return embedding    

def verify_model():
    vector = Vector()
    
    print(f"Model loaded: {vector.model}")
    print(f"Max sequence length: {vector.model.max_seq_length}")

def embed_text(text):
    vector = Vector()
    embedding = vector.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")