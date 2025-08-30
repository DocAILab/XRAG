import os
import glob
import time
import torch
import faiss
import pickle
import numpy as np
import transformers
from ..config import Config
from ..utils import get_module_logger
from typing import List, Dict, Optional, Any

logger = get_module_logger(__name__)


class SelfRAGRetriever:
    """
    Provides document retrieval capabilities for Self-RAG models.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Self-RAG retriever.

        Params:
            config (Config, optional): Configuration object. If None, uses global config.
        """
        self.config = config or Config()
        self.self_rag_config = self.config.config.get("self_rag", {})

        # Model
        self.model = None
        self.tokenizer = None
        self.index = None
        self.passages = None
        self.passage_id_map = None

        self.retriever_model = self.self_rag_config.get(
            "retriever_model", "facebook/contriever-msmarco"
        )
        self.max_length = 512
        self.batch_size = 64
        self.n_docs = 100

        self.cuda_available = torch.cuda.is_available()

        self._load_retriever_model()

        logger.info(f"Initialized SelfRAGRetriever with model: {self.retriever_model}")

    def _load_retriever_model(self):
        """Load the retriever model and tokenizer."""
        try:
            logger.info(f"Loading retriever model: {self.retriever_model}")

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.retriever_model
            )
            self.model = transformers.AutoModel.from_pretrained(self.retriever_model)

            if self.cuda_available:
                self.model = self.model.cuda()
                self.model = self.model.half()

            self.model.eval()
            logger.info("Retriever model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load retriever model: {e}")
            raise

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Embed a list of queries using the retriever model.

        Params:
            queries (List[str]): List of query strings

        Returns:
            Numpy array of query embeddings
        """
        if self.model is None or self.tokenizer is None:
            self._load_retriever_model()

        embeddings = []
        batch_queries = []

        with torch.no_grad():
            for k, query in enumerate(queries):
                batch_queries.append(query)

                if len(batch_queries) == self.batch_size or k == len(queries) - 1:
                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_queries,
                        return_tensors="pt",
                        max_length=self.max_length,
                        padding=True,
                        truncation=True,
                    )

                    if self.cuda_available:
                        encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}

                    # Get embeddings
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())

                    batch_queries = []

        embeddings = torch.cat(embeddings, dim=0)
        logger.debug(f"Query embeddings shape: {embeddings.size()}")
        return embeddings.numpy()

    def setup_index(self, passages_path: str, embeddings_path: str = None):
        """
        Setup the retrieval index and load passages.

        Params:
            passages_path (str): Path to passages file (.jsonl or .tsv)
            embeddings_path (str, optional): Path to precomputed embeddings
        """
        try:
            # Load passages
            logger.info(f"Loading passages from: {passages_path}")
            self.passages = self._load_passages(passages_path)
            self.passage_id_map = {p["id"]: p for p in self.passages}
            logger.info(f"Loaded {len(self.passages)} passages")

            # Setup index
            if embeddings_path and os.path.exists(embeddings_path):
                self._load_precomputed_index(embeddings_path)
            else:
                logger.warning("No precomputed embeddings found. Index not available.")
                self.index = None

        except Exception as e:
            logger.error(f"Failed to setup index: {e}")
            raise

    def _load_passages(self, passages_path: str) -> List[Dict[str, Any]]:
        """Load passages from file."""
        passages = []

        if passages_path.endswith(".jsonl"):
            import json

            with open(passages_path, "r", encoding="utf-8") as f:
                for line in f:
                    passage = json.loads(line.strip())
                    passages.append(passage)
        elif passages_path.endswith(".tsv"):
            import csv

            with open(passages_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for i, row in enumerate(reader):
                    # Convert TSV format to expected format
                    passage = {
                        "id": row.get("title", str(i)),
                        "title": row.get("title", ""),
                        "text": row.get("text", ""),
                    }
                    passages.append(passage)
        else:
            raise ValueError(f"Unsupported file format: {passages_path}")

        return passages

    def _load_precomputed_index(self, embeddings_path: str):
        """Load precomputed embeddings and setup FAISS index."""
        try:
            # Load embeddings
            embedding_files = glob.glob(embeddings_path)
            if not embedding_files:
                logger.warning(f"No embedding files found at: {embeddings_path}")
                return

            all_embeddings = []
            all_ids = []

            for file_path in sorted(embedding_files):
                logger.debug(f"Loading embeddings from: {file_path}")
                with open(file_path, "rb") as f:
                    ids, embeddings = pickle.load(f)
                    all_embeddings.append(embeddings)
                    all_ids.extend(ids)

            all_embeddings = np.vstack(all_embeddings)

            dimension = all_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(all_embeddings.astype("float32"))

            logger.info(f"Created FAISS index with {self.index.ntotal} vectors")

        except Exception as e:
            logger.error(f"Failed to load precomputed index: {e}")
            self.index = None

    def search_documents(self, query: str, n_docs: int = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents given a query.

        Params:
            query (str): Search query
            n_docs (int, optional): Number of documents to retrieve (default from config)

        Returns:
            List of retrieved documents with metadata
        """
        if n_docs is None:
            n_docs = self.n_docs

        if self.index is None:
            logger.warning("No index available. Cannot perform retrieval.")
            return []

        try:
            # Embed the query
            query_embedding = self.embed_queries([query])

            # Search in the index
            start_time = time.time()
            indices, scores = self.index.search_knn(
                query_embedding.astype("float32"), n_docs
            )
            search_time = time.time() - start_time

            logger.debug(f"Search completed in {search_time:.2f}s")

            # Retrieve documents
            retrieved_docs = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.passages):
                    doc = self.passages[idx].copy()
                    doc["score"] = float(score)
                    doc["rank"] = i + 1
                    retrieved_docs.append(doc)

            return retrieved_docs

        except Exception as e:
            logger.error(f"Error during document search: {e}")
            return []
