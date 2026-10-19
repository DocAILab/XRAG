from llama_index.core import VectorStoreIndex
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore

from ..data.qa_loader import get_documents
import os
from pathlib import Path
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from typing import Any, Callable, Dict, Optional


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


class _EmbeddingProgressHandler(BaseCallbackHandler):
    def __init__(self, total: int, callback: ProgressCallback):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.total = total
        self.completed = 0
        self.callback = callback

    def on_event_start(self, event_type, payload=None, event_id="", parent_id="", **kwargs):
        return event_id

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        if event_type != CBEventType.EMBEDDING or not self.callback:
            return
        self.completed = min(self.total, self.completed + len((payload or {}).get(EventPayload.CHUNKS, [])))
        self.callback({
            "phase": "embedding",
            "completed": self.completed,
            "total": self.total,
            "progress": 0.2 + 0.75 * self.completed / max(1, self.total),
        })

    def start_trace(self, trace_id=None):
        pass

    def end_trace(self, trace_id=None, trace_map=None):
        pass


def _report(callback: ProgressCallback, **payload: Any) -> None:
    if callback:
        callback(payload)


def load_persisted_index(persist_dir, split_type="sentence", progress_callback: ProgressCallback = None):
    path = Path(persist_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Vector database directory does not exist: {path}")

    required_files = ("docstore.json", "index_store.json", "default__vector_store.json")
    missing = [name for name in required_files if not (path / name).is_file()]
    if missing:
        raise ValueError(
            "Invalid vector database directory; missing: " + ", ".join(missing)
        )

    _report(progress_callback, phase="loading_existing", completed=0, total=1, progress=0.5)
    hierarchical_storage_context = None
    if split_type == "hierarchical":
        hierarchical_path = Path(f"{path}-hierarchical")
        if not hierarchical_path.is_dir():
            raise FileNotFoundError(
                f"Hierarchical storage directory does not exist: {hierarchical_path}"
            )
        hierarchical_storage_context = StorageContext.from_defaults(
            persist_dir=str(hierarchical_path)
        )
    storage_context = StorageContext.from_defaults(persist_dir=str(path))
    index = load_index_from_storage(storage_context)
    _report(progress_callback, phase="done", completed=1, total=1, progress=1.0)
    return index, hierarchical_storage_context


def get_index(documents, persist_dir, split_type="sentence", chunk_size=1024,chunk_overlap=20,chunk_sizes=[2048, 512, 128],semantic_setting=None, window_size=3, progress_callback: ProgressCallback = None):
    hierarchical_storage_context = None
    if not os.path.exists(persist_dir):
        # load the documents and create the index
        _report(progress_callback, phase="splitting", completed=0, total=len(documents), progress=0.1)
        if split_type == "sentence":
            parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            print("nodes: " + str(nodes.__len__()))
        elif split_type == "sentence_window":
            parser = SentenceWindowNodeParser.from_defaults(
                window_size=window_size,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            print("nodes: " + str(nodes.__len__()))
        elif split_type == "character":
            parser = LangchainNodeParser(RecursiveCharacterTextSplitter())
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            print("nodes: " + str(nodes.__len__()))
        elif split_type == "hierarchical":
            parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=chunk_sizes
            )
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            print("nodes: " + str(nodes.__len__()))
        elif split_type == "semantic":
            parser = SemanticSplitterNodeParser(
                buffer_size=semantic_setting["buffer_size"],
                embed_model=Settings.embed_model,
                include_metadata=semantic_setting["include_metadata"],
                include_prev_next_rel=semantic_setting["include_prev_next_rel"],
                breakpoint_percentile_threshold=semantic_setting["breakpoint_percentile_threshold"]
            )
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            print("nodes: " + str(nodes.__len__()))
        else:
            raise ValueError(f"split_type {split_type} not supported.")
        _report(progress_callback, phase="embedding", completed=0, total=len(nodes), progress=0.2)
        progress_handler = _EmbeddingProgressHandler(len(nodes), progress_callback)
        Settings.embed_model.callback_manager.add_handler(progress_handler)
        try:
            index = VectorStoreIndex(nodes, show_progress=True)
        finally:
            Settings.embed_model.callback_manager.remove_handler(progress_handler)
        _report(progress_callback, phase="persisting", completed=len(nodes), total=len(nodes), progress=0.96)
        # store it for later
        if split_type == "hierarchical":
            docstore = SimpleDocumentStore()
            docstore.add_documents(nodes)
            hierarchical_storage_context = StorageContext.from_defaults(docstore=docstore)
            # save
            hierarchical_storage_context.persist(persist_dir=persist_dir+"-hierarchical")

        index.storage_context.persist(persist_dir=persist_dir)
    else:
        # load the existing index
        return load_persisted_index(
            persist_dir, split_type=split_type, progress_callback=progress_callback
        )
    _report(progress_callback, phase="done", completed=1, total=1, progress=1.0)
    return index, hierarchical_storage_context
