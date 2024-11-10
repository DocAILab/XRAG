import streamlit as st
import pandas as pd
import plotly.express as px
from ragx.config import Config
from ragx.launcher import run
from ragx.eval.evaluate_rag import EvaluationResult

AVAILABLE_METRICS = [
    "NLG_chrf", "NLG_bleu", "NLG_meteor", "NLG_wer", "NLG_cer", "NLG_chrf_pp",
    "NLG_mauve", "NLG_perplexity",
    "NLG_rouge_rouge1", "NLG_rouge_rouge2", "NLG_rouge_rougeL", "NLG_rouge_rougeLsum",
    "Llama_retrieval_Faithfulness", "Llama_retrieval_Relevancy", "Llama_response_correctness",
    "Llama_response_semanticSimilarity", "Llama_response_answerRelevancy", "Llama_retrieval_RelevancyG",
    "Llama_retrieval_FaithfulnessG",
    "DeepEval_retrieval_contextualPrecision", "DeepEval_retrieval_contextualRecall",
    "DeepEval_retrieval_contextualRelevancy", "DeepEval_retrieval_faithfulness",
    "DeepEval_response_answerRelevancy", "DeepEval_response_hallucination",
    "DeepEval_response_bias", "DeepEval_response_toxicity",
    "UpTrain_Response_Completeness", "UpTrain_Response_Conciseness", "UpTrain_Response_Relevance",
    "UpTrain_Response_Valid", "UpTrain_Response_Consistency", "UpTrain_Response_Response_Matching",
    "UpTrain_Retrieval_Context_Relevance", "UpTrain_Retrieval_Context_Utilization",
    "UpTrain_Retrieval_Factual_Accuracy", "UpTrain_Retrieval_Context_Conciseness",
    "UpTrain_Retrieval_Code_Hallucination",
]

# Define options for each dropdown
LLM_OPTIONS = ["chatgpt-3.5", "gpt-4", "text-davinci-003"]  # Add more as needed
EMBEDDING_OPTIONS = ["BAAI/bge-large-en-v1.5", "openai"]  # Add more as needed
SPLIT_TYPE_OPTIONS = ["sentence", "character", "hierarchical"]
DATASET_OPTIONS = ["hotpot_qa", "drop", "natural_questions","trivia_qa","search_qa","finqa","law"]  # Replace with actual dataset options
RETRIEVER_OPTIONS = ["BM25", "Vector", "Summary", "Tree", "Keyword", "Custom", "QueryFusion", "AutoMerging", "Recursive", "SentenceWindow"]  # Add more as needed
POSTPROCESS_RERANK_OPTIONS = ["none","long_context_reorder", "colbertv2_rerank","bge-reranker-base"]  # Add more as needed
QUERY_TRANSFORM_OPTIONS = ["none", "hyde_zeroshot", "hyde_fewshot","stepback_zeroshot","stepback_fewshot"]  # Add more as needed

def main():
    st.set_page_config(layout="wide")
    st.title("RAG Configuration and Evaluation Interface")

    cfg = Config()

    col1, col2 = st.columns([1, 3])
    st.header("Configuration Parameters")

    with col1:


        # API Keys
        st.subheader("API Keys")
        cfg.api_key = st.text_input("API Key", value=cfg.api_key, type="password")
        cfg.api_base = st.text_input("API Base", value=cfg.api_base)
        cfg.api_name = st.text_input("API Name", value=cfg.api_name)
        cfg.auth_token = st.text_input("Auth Token", value=cfg.auth_token)

    with col2:

        # Settings
        st.subheader("Settings")
        c1, c2, c3 = st.columns([1, 1,1])
        with c1:

            cfg.llm = st.selectbox("LLM", options=LLM_OPTIONS, index=LLM_OPTIONS.index(cfg.llm) if cfg.llm in LLM_OPTIONS else 0)
            cfg.embeddings = st.selectbox("Embeddings", options=EMBEDDING_OPTIONS, index=EMBEDDING_OPTIONS.index(cfg.embeddings) if cfg.embeddings in EMBEDDING_OPTIONS else 0)
            cfg.split_type = st.selectbox("Split Type", options=SPLIT_TYPE_OPTIONS, index=SPLIT_TYPE_OPTIONS.index(cfg.split_type))
            cfg.chunk_size = st.number_input("Chunk Size", min_value=1, value=cfg.chunk_size, step=1)
            cfg.dataset = st.selectbox("Dataset", options=DATASET_OPTIONS, index=DATASET_OPTIONS.index(cfg.dataset) if cfg.dataset in DATASET_OPTIONS else 0)
            cfg.source_dir = st.text_input("Source Directory", value=cfg.source_dir)
            cfg.persist_dir = st.text_input("Persist Directory", value=cfg.persist_dir)

        with c2:

            cfg.test_init_total_number_documents = st.number_input("Test Initial Total Number of Documents", min_value=1, value=cfg.test_init_total_number_documents, step=1)
            cfg.retriever = st.selectbox("Retriever", options=RETRIEVER_OPTIONS, index=RETRIEVER_OPTIONS.index(cfg.retriever) if cfg.retriever in RETRIEVER_OPTIONS else 0)
            cfg.retriever_mode = st.selectbox("Retriever Mode", options=[0, 1], index=cfg.retriever_mode)
            cfg.postprocess_rerank = st.selectbox("Postprocess Rerank", options=POSTPROCESS_RERANK_OPTIONS, index=POSTPROCESS_RERANK_OPTIONS.index(cfg.postprocess_rerank) if cfg.postprocess_rerank in POSTPROCESS_RERANK_OPTIONS else 0)
            cfg.query_transform = st.selectbox("Query Transform", options=QUERY_TRANSFORM_OPTIONS, index=QUERY_TRANSFORM_OPTIONS.index(cfg.query_transform) if cfg.query_transform in QUERY_TRANSFORM_OPTIONS else 0)

        # Experiment Flag
        with c3:
            cfg.experiment_1 = st.checkbox("Experiment 1", value=cfg.experiment_1)
            cfg.n = st.number_input("Number of Documents (n)", min_value=1, value=cfg.n, step=1)

        # Evaluation Metrics Selection
    st.subheader("Evaluation Metrics")
    cfg.metrics = st.multiselect("Select Evaluation Metrics", options=AVAILABLE_METRICS, default=AVAILABLE_METRICS[:5])

    # Button to Run
    if st.button("Run Evaluation"):
        with st.spinner("Running evaluation..."):
            evaluation_results = run()
        st.success("Evaluation complete!")
        st.session_state.evaluation_results = evaluation_results


    st.header("Evaluation Results")
    if 'evaluation_results' in st.session_state:
        results = st.session_state.evaluation_results
        display_results(results)

def display_results(results: EvaluationResult):
    # Display summary statistics
    st.subheader("Summary Statistics")
    summary = pd.DataFrame(results.get_summary(), index=[0])
    st.table(summary)

    # Display detailed metrics
    st.subheader("Detailed Metrics")
    metrics_df = pd.DataFrame(results.get_all_metrics())
    st.dataframe(metrics_df)

    # Visualize metrics
    st.subheader("Metric Visualization")
    metric_to_plot = st.selectbox("Select a metric to visualize", options=results.metrics)
    fig = px.box(metrics_df, y=metric_to_plot, title=f"Distribution of {metric_to_plot}")
    st.plotly_chart(fig)

    # Display sample evaluations
    st.subheader("Sample Evaluations")
    num_samples = st.slider("Number of samples to display", min_value=1, max_value=len(results.evaluations), value=5)
    for i, eval_result in enumerate(results.evaluations[:num_samples]):
        st.write(f"Sample {i+1}")
        st.json(eval_result)

if __name__ == "__main__":
    main()
