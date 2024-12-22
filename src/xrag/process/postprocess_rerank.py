from llama_index.core.postprocessor import LongContextReorder
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

# !pip install llama-index-postprocessor-colbert-rerank
# !pip install llama-index-postprocessor-cohere-rerank

def get_postprocessor(cfg):
    # postprocess rerank, available: long_context_reorder, colbertv2_rerank, cohere_rerank, bge-reranker-base
    if cfg.postprocess_rerank == 'long_context_reorder':
        return LongContextReorder()
    elif cfg.postprocess_rerank == 'colbertv2_rerank':
        return ColbertRerank()
    elif cfg.postprocess_rerank == 'cohere_rerank':
        return CohereRerank()
    elif cfg.postprocess_rerank == 'bge-reranker-base':
        return FlagEmbeddingReranker(model="BAAI/bge-reranker-base")
    else:
        raise Exception("postprocess_rerank not supported: %s" % cfg.postprocess_rerank)