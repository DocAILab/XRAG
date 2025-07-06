from llama_index.core import VectorStoreIndex
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore

from ..data.qa_loader import get_documents
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceWindowNodeParser

def get_index(documents, persist_dir, split_type="sentence", chunk_size=1024,chunk_overlap=20,chunk_sizes=[2048, 512, 128],semantic_setting=None, window_size=3):
    hierarchical_storage_context = None
    if not os.path.exists(persist_dir):
        # load the documents and create the index
        if split_type == "sentence":
            parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            print("nodes: " + str(nodes.__len__()))
            index = VectorStoreIndex(nodes,show_progress=True)
        elif split_type == "sentence_window":
            parser = SentenceWindowNodeParser.from_defaults(
                window_size=window_size,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            print("nodes: " + str(nodes.__len__()))
            index = VectorStoreIndex(nodes, show_progress=True)
        elif split_type == "character":
            parser = LangchainNodeParser(RecursiveCharacterTextSplitter())
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            print("nodes: " + str(nodes.__len__()))
            index = VectorStoreIndex(nodes,show_progress=True)
        elif split_type == "hierarchical":
            parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=chunk_sizes
            )
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            print("nodes: " + str(nodes.__len__()))
            index = VectorStoreIndex(nodes,show_progress=True)
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
            index = VectorStoreIndex(nodes,show_progress=True)
        else:
            raise ValueError(f"split_type {split_type} not supported.")
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
        if split_type == "hierarchical":
            hierarchical_storage_context = StorageContext.from_defaults(persist_dir=persist_dir + "-hierarchical")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    return index, hierarchical_storage_context
