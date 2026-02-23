from sentence_transformers import SentenceTransformer

from src.lib.utils import vector_length, vector_magnitude

class Vector:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    @staticmethod
    def add_vectors(vec1, vec2):
        
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")
        
        return [vec1[i] + vec2[i] for i in range(len(vec1))]

    @staticmethod
    def subtract_vectors(vec1, vec2):

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
    

    def dot_product(self, vec1, vec2):

        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")

        magnitude_vec1 = vector_magnitude(vec1)
        magnitude_vec2 = vector_magnitude(vec2)

        if magnitude_vec1 == 0 or magnitude_vec2 == 0:
            return 0.0
        
        dot_product = self.dot_product(vec1, vec2)
        
        return sum
