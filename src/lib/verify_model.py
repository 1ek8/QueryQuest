from src.embedding.vector import Vector


def verify_model():
    vector = Vector()
    
    print(f"Model loaded: {vector.model}")
    print(f"Max sequence length: {vector.model.max_seq_length}")

