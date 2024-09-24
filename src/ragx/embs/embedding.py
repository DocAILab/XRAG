from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.legacy.embeddings import HuggingFaceEmbedding

def get_embedding(name):
    return HuggingFaceEmbedding(
        model_name=name,
        embed_batch_size=16,
        # cache_folder="./embedding_model"
    )

'''
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def get_embedding(name):
    encode_kwargs = {"batch_size": 128, 'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(
        model_name=name,
        encode_kwargs=encode_kwargs,
        # embed_batch_size=128,
    )
    return embeddings
'''