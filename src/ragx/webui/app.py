import warnings

import streamlit as st
import pandas as pd
import plotly.express as px
from ragx.config import Config
from ragx.eval.EvalModelAgent import EvalModelAgent
from ragx.eval.evaluate_LLM import evaluating
from ragx.launcher import run
from ragx.eval.evaluate_rag import EvaluationResult
from ragx.process.query_transform import transform_and_query
from ragx.launcher import build_index, build_query_engine
from ragx.data.qa_loader import get_qa_dataset

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

@st.cache_resource(show_spinner=False)
def get_query():
    return run(cli=False)

@st.cache_resource(show_spinner=False)
def get_qa_dataset_(dataset):
    return get_qa_dataset(dataset)

@st.cache_resource(show_spinner=False)
def get_index():
    return build_index(st.session_state.qa_dataset)

@st.cache_resource(show_spinner=False)
def get_query_engine():
    return build_query_engine(st.session_state.index, st.session_state.hierarchical_storage_context)


def main():

    if "step" not in st.session_state:
        st.session_state.step = 1
    st.set_page_config(layout="wide")
    st.title("XRAG")
    cfg = Config()

    if st.session_state.step == 1:


        st.header("Choose your Dataset")
        cfg.dataset = st.selectbox("Dataset", options=DATASET_OPTIONS,
                                   index=DATASET_OPTIONS.index(cfg.dataset) if cfg.dataset in DATASET_OPTIONS else 0,key="dataset")
        if st.button("Load Dataset"):
            st.session_state.step = 2
            with st.spinner("Loading Dataset..."):
                st.session_state.qa_dataset = get_qa_dataset_(cfg.dataset)
            st.rerun()

    if st.session_state.step == 2:
        st.header("Configure your RAG Index")
        st.markdown("Selected Dataset: " + cfg.dataset)
        col1, col2 = st.columns([1, 1])
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

            cfg.llm = st.selectbox("LLM", options=LLM_OPTIONS, index=LLM_OPTIONS.index(cfg.llm) if cfg.llm in LLM_OPTIONS else 0)
            cfg.embeddings = st.selectbox("Embeddings", options=EMBEDDING_OPTIONS, index=EMBEDDING_OPTIONS.index(cfg.embeddings) if cfg.embeddings in EMBEDDING_OPTIONS else 0)
            cfg.split_type = st.selectbox("Split Type", options=SPLIT_TYPE_OPTIONS, index=SPLIT_TYPE_OPTIONS.index(cfg.split_type))
            cfg.chunk_size = st.number_input("Chunk Size", min_value=1, value=cfg.chunk_size, step=1)
            cfg.source_dir = st.text_input("Source Directory", value=cfg.source_dir)
            cfg.persist_dir = st.text_input("Persist Directory", value=cfg.persist_dir)

        if st.button("Build Index"):
            st.session_state.step = 3
            with st.spinner("Building Index..."):
                st.session_state.index, st.session_state.hierarchical_storage_context = get_index()
            st.rerun()
            # Evaluation Metrics Selection

    if st.session_state.step == 3:
        st.header("Configure your RAG Query Engine")
        cfg.retriever = st.selectbox("Retriever", options=RETRIEVER_OPTIONS, index=RETRIEVER_OPTIONS.index(
            cfg.retriever) if cfg.retriever in RETRIEVER_OPTIONS else 0)
        cfg.retriever_mode = st.selectbox("Retriever Mode", options=[0, 1], index=cfg.retriever_mode)
        cfg.postprocess_rerank = st.selectbox("Postprocess Rerank", options=POSTPROCESS_RERANK_OPTIONS,
                                              index=POSTPROCESS_RERANK_OPTIONS.index(
                                                  cfg.postprocess_rerank) if cfg.postprocess_rerank in POSTPROCESS_RERANK_OPTIONS else 0)
        cfg.query_transform = st.selectbox("Query Transform", options=QUERY_TRANSFORM_OPTIONS,
                                           index=QUERY_TRANSFORM_OPTIONS.index(
                                               cfg.query_transform) if cfg.query_transform in QUERY_TRANSFORM_OPTIONS else 0)
        cfg.query_transform_mode = st.selectbox("Query Transform Mode", options=[0, 1], index=cfg.query_transform_mode)
        cfg.text_qa_template_str = st.text_area("Text QA Template", value=cfg.text_qa_template_str)
        cfg.refine_template_str = st.text_area("Refine Template", value=cfg.refine_template_str)

        if st.button("Build Query Engine"):
            st.session_state.step = 4
            with st.spinner("Building Query Engine..."):
                st.session_state.query_engine = get_query_engine()
            st.rerun()
    if st.session_state.step == 4:
        st.header("Evaluate your RAG Model with single question")
        prompt = st.text_input('Input your question here')

        # If the user hits enter
        if prompt:
            response = transform_and_query(prompt, cfg, st.session_state.query_engine)
            st.write(response.response)

            # Display source text
            with st.expander('Source Text'):
                st.write(response.get_formatted_sources(length=1024))

        if st.button("Evaluate Your Dataset"):
            st.session_state.step = 5
            st.rerun()

    if st.session_state.step == 5:
        st.header("Evaluate your RAG Model with your dataset")
        cfg.metrics = st.multiselect("Evaluation Metrics", options=AVAILABLE_METRICS, default=AVAILABLE_METRICS)
        # cfg.n = st.number_input("Number of documents to evaluate", min_value=1, value=cfg.n, step=1)
        cfg.test_init_total_number_documents = st.number_input("Total number of documents to evaluate", min_value=1, value=cfg.test_init_total_number_documents, step=1)

        if st.button("Evaluate Your Dataset"):

            all_num = 0
            evaluateResults = EvaluationResult(metrics=cfg.metrics)
            evalAgent = EvalModelAgent(cfg)
            if cfg.experiment_1:
                if len(st.session_state.qa_dataset) < cfg.test_init_total_number_documents:
                    warnings.filterwarnings('default')
                    warnings.warn("使用的数据集长度大于数据集本身的最大长度，请修改。 本轮代码无法运行", UserWarning)
            else:
                cfg.test_init_total_number_documents = cfg.n
            for question, expected_answer, golden_context, golden_context_ids in zip(
                    st.session_state.qa_dataset['test_data']['question'][:cfg.test_init_total_number_documents],
                    st.session_state.qa_dataset['test_data']['expected_answer'][:cfg.test_init_total_number_documents],
                    st.session_state.qa_dataset['test_data']['golden_context'][:cfg.test_init_total_number_documents],
                    st.session_state.qa_dataset['test_data']['golden_context_ids'][:cfg.test_init_total_number_documents]
            ):
                response = transform_and_query(question, cfg, st.session_state.query_engine)
                # 返回node节点
                retrieval_ids = []
                retrieval_context = []
                for source_node in response.source_nodes:
                    retrieval_ids.append(source_node.metadata['id'])
                    retrieval_context.append(source_node.get_content())
                actual_response = response.response
                eval_result = evaluating(question, response, actual_response, retrieval_context, retrieval_ids,
                                         expected_answer, golden_context, golden_context_ids, evaluateResults.metrics,
                                         evalAgent)
                with st.expander(question):
                    st.markdown("### Answer")
                    st.markdown(response.response)
                    st.markdown('### Retrieval context')
                    st.markdown('\n\n'.join(retrieval_context))
                    st.markdown('### Golden context')


                evaluateResults.add(eval_result)
                all_num = all_num + 1
                st.markdown(evaluateResults.get_results_str())

                # print("总数：" + str(all_num))
            st.success("Evaluation complete!")
            st.session_state.evaluation_results = evaluateResults
            # st.session_state.step = 5
            st.rerun()

    # if st.session_state.step == 5:
    #     st.header("Evaluation Results")
    #     if 'evaluation_results' in st.session_state:
    #         results = st.session_state.evaluation_results
    #         display_results(results)

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
