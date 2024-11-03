import streamlit as st
from ..config import Config  # Adjust the import based on your project structure
from ..launcher import run  # Adjust the import based on your project structure

def main():
    st.title("RAG Configuration Interface")

    cfg = Config()

    st.sidebar.header("Configuration Parameters")

    # API Keys
    st.sidebar.subheader("API Keys")
    api_key = st.sidebar.text_input("API Key", value=cfg.api_key)
    api_base = st.sidebar.text_input("API Base", value=cfg.api_base)
    api_name = st.sidebar.text_input("API Name", value=cfg.api_name)
    auth_token = st.sidebar.text_input("Auth Token", value=cfg.auth_token)

    # Settings
    st.sidebar.subheader("Settings")
    llm = st.sidebar.text_input("LLM", value=cfg.llm)
    embeddings = st.sidebar.text_input("Embeddings", value=cfg.embeddings)
    split_type = st.sidebar.selectbox("Split Type", options=["sentence", "word", "paragraph"], index=["sentence", "word", "paragraph"].index(cfg.split_type))
    chunk_size = st.sidebar.number_input("Chunk Size", min_value=1, value=cfg.chunk_size, step=1)
    dataset = st.sidebar.text_input("Dataset", value=cfg.dataset)
    source_dir = st.sidebar.text_input("Source Directory", value=cfg.source_dir)
    persist_dir = st.sidebar.text_input("Persist Directory", value=cfg.persist_dir)
    n = st.sidebar.number_input("Number of Documents (n)", min_value=1, value=cfg.n, step=1)
    test_init_total_number_documents = st.sidebar.number_input("Test Initial Total Number of Documents", min_value=1, value=cfg.test_init_total_number_documents, step=1)
    retriever = st.sidebar.text_input("Retriever", value=cfg.retriever)
    retriever_mode = st.sidebar.selectbox("Retriever Mode", options=[0, 1], index=[0, 1].index(cfg.retriever_mode))
    postprocess_rerank = st.sidebar.text_input("Postprocess Rerank", value=cfg.postprocess_rerank)
    query_transform = st.sidebar.text_input("Query Transform", value=cfg.query_transform)

    # Experiment Flag
    experiment_1 = st.sidebar.checkbox("Experiment 1", value=cfg.experiment_1)

    # Button to Run
    if st.button("Run"):
        # Collect overrides
        overrides = {
            "api_key": api_key,
            "api_base": api_base,
            "api_name": api_name,
            "auth_token": auth_token,
            "llm": llm,
            "embeddings": embeddings,
            "split_type": split_type,
            "chunk_size": chunk_size,
            "dataset": dataset,
            "source_dir": source_dir,
            "persist_dir": persist_dir,
            "n": n,
            "test_init_total_number_documents": test_init_total_number_documents,
            "retriever": retriever,
            "retriever_mode": retriever_mode,
            "postprocess_rerank": postprocess_rerank,
            "query_transform": query_transform,
            "experiment_1": experiment_1
        }

        # Update the Config instance
        cfg.update_config(overrides)

        # Run the main function
        run()

if __name__ == "__main__":
    main()



