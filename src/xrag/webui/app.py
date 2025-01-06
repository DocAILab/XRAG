import warnings
from streamlit_card import card
import streamlit as st
import pandas as pd
import plotly.express as px
from xrag.config import Config
from xrag.eval.EvalModelAgent import EvalModelAgent
from xrag.eval.evaluate_rag import evaluating
from xrag.launcher import run
from xrag.eval.evaluate_rag import EvaluationResult
from xrag.process.query_transform import transform_and_query
from xrag.launcher import build_index, build_query_engine
from xrag.data.qa_loader import get_qa_dataset

AVAILABLE_METRICS = [
    "NLG_chrf", "NLG_meteor", "NLG_wer", "NLG_cer", "NLG_chrf_pp",
    "NLG_perplexity",
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
LLM_OPTIONS = [
    # huggingface models
    # 'llama',      # meta-llama/Llama-2-7b-chat-hf
    # 'chatglm',    # THUDM/chatglm3-6b
    # 'qwen',       # Qwen/Qwen1.5-7B-Chat
    # 'qwen14_int8',# Qwen/Qwen1.5-14B-Chat-GPTQ-Int8
    # 'qwen7_int8', # Qwen/Qwen1.5-7B-Chat-GPTQ-Int8
    # 'qwen1.8',    # Qwen/Qwen1.5-1.8B-Chat
    # 'baichuan',   # baichuan-inc/Baichuan2-7B-Chat
    # 'falcon',     # tiiuae/falcon-7b-instruct
    # 'mpt',        # mosaicml/mpt-7b-chat
    # 'yi',         # 01-ai/Yi-6B-Chat

    'openai',     # OpenAI API
    'huggingface',# HuggingFace local models
    'ollama',     # Ollama local models
]

HF_MODEL_OPTIONS = [
    'llama',      # meta-llama/Llama-2-7b-chat-hf
    'chatglm',    # THUDM/chatglm3-6b
    'qwen',       # Qwen/Qwen1.5-7B-Chat
    'qwen14_int8',# Qwen/Qwen1.5-14B-Chat-GPTQ-Int8
    'qwen7_int8', # Qwen/Qwen1.5-7B-Chat-GPTQ-Int8
    'qwen1.8',    # Qwen/Qwen1.5-1.8B-Chat
    'baichuan',   # baichuan-inc/Baichuan2-7B-Chat
    'falcon',     # tiiuae/falcon-7b-instruct
    'mpt',        # mosaicml/mpt-7b-chat
    'yi',         # 01-ai/Yi-6B-Chat
]

# ollama cascade dict
OLLAMA_OPTIONS = {
    "LLaMA": {
        "llama2-7b": "llama2:7b",
        "llama2-13b": "llama2:13b",
        "llama2-70b": "llama2:70b",

        "llama3.1-8b": "llama3.1:8b",
        "llama3.1-70b": "llama3.1:70b",

        "llama3.2-1b": "llama3.2:1b",
        "llama3.2-3b": "llama3.2:3b",

    },
    "Mistral": {
        "mistral-7b": "mistral:7b",
        "mixtral": "mixtral",
        "mistral-7b-q4": "mistral:7b-q4_0",
    },
    "others": {
        "gemma-7b": "gemma:7b",
        "codellama": "codellama",
        "neural-chat": "neural-chat",
        "other": "other"
    },
}

EMBEDDING_OPTIONS = ["BAAI/bge-large-en-v1.5", "BAAI/bge-m3", "BAAI/bge-base-en-v1.5","BAAI/bge-small-en-v1.5","BAAI/bge-large-zh-v1.5","BAAI/bge-base-zh-v1.5","BAAI/bge-small-zh-v1.5"]  # Add more as needed
SPLIT_TYPE_OPTIONS = ["sentence", "character", "hierarchical"]
DATASET_OPTIONS = ["hotpot_qa", "drop", "natural_questions","trivia_qa","search_qa","finqa","law"]  # Replace with actual dataset options
DATASET_DISPLAY_MAP = {
    "hotpot_qa": "HotpotQA",
    "drop": "DropQA",
    "natural_questions": "NaturalQA"
    # remain for other options
}

DATASETS_INFO = {
    "HotpotQA": {
        "size": {
            "train": "86,830",
            "validation": "8,680",
            "test": "968"
        },
        "corpus": {
            "documents": "508,826",
            "source": "Wikipedia"
        },
        "features": {
            "Multi-hop": True,
            "Constrained": False,
            "Numerical": True,
            "Set-logical": False
        }
    },
    "DropQA": {
        "size": {
            "train": "78,241",
            "validation": "7,824",
            "test": "870"
        },
        "corpus": {
            "documents": "6,147",
            "source": "Wikipedia"
        },
        "features": {
            "Multi-hop": True,
            "Constrained": False,
            "Numerical": True,
            "Set-logical": True
        }
    },
    "NaturalQA": {
        "size": {
            "train": "100,093",
            "validation": "10,010",
            "test": "1,112"
        },
        "corpus": {
            "documents": "49,815",
            "source": "Wikipedia"
        },
        "features": {
            "Multi-hop": True,
            "Constrained": True,
            "Numerical": True,
            "Set-logical": False
        }
    }
}

# frontend dataset options
FRONTEND_DATASET_OPTIONS = [DATASET_DISPLAY_MAP.get(ds, ds) for ds in DATASET_OPTIONS]
RETRIEVER_OPTIONS = ["BM25", "Vector", "Summary", "Tree", "Keyword", "Custom", "QueryFusion", "AutoMerging", "Recursive", "SentenceWindow"]  # Add more as needed
POSTPROCESS_RERANK_OPTIONS = ["none","long_context_reorder", "colbertv2_rerank","bge-reranker-base"]  # Add more as needed
QUERY_TRANSFORM_OPTIONS = ["none", "hyde_zeroshot", "hyde_fewshot","stepback_zeroshot","stepback_fewshot"]  # Add more as needed

# follow the order listed in the docs
METRIC_DISPLAY_MAP = {
    "NLG_chrf": "ChrF",
    "NLG_chrf_pp": "ChrF++",
    "NLG_meteor": "METEOR",
    "NLG_rouge_rouge1": "ROUGE1",
    "NLG_rouge_rouge2": "ROUGE2",
    "NLG_rouge_rougeL": "ROUGEL",
    "NLG_rouge_rougeLsum": "ROUGELSUM",
    "nlg-em": "EM",
    "NLG_perplexity": "PPL",
    "NLG_cer": "CER",
    "NLG_wer": "WER",
    
    # Llama metrics
    "Llama_retrieval_Faithfulness": "Llama-Response-Faithfulness",
    "Llama_retrieval_Relevancy": "Llama-Response-Relevance",
    "Llama_response_correctness": "Llama-Response-Correctness",
    "Llama_response_semanticSimilarity": "Llama-Response-Similarity",
    "Llama_response_answerRelevancy": "Llama-Response-Relevance++",
    "Llama_retrieval_FaithfulnessG": "Llama-Response-Faithfulness+",
    "Llama_retrieval_RelevancyG": "Llama-Response-Relevance+",
    
    # UpTrain metrics
    "UpTrain_Retrieval_Context_Relevance": "Uptrain-Context-Relevance",
    "UpTrain_Retrieval_Context_Conciseness": "Uptrain-Context-Conciseness",
    "DeepEval_response_answerRelevancy": "DeepEval-Response-Relevancy",
    "UpTrain_Response_Completeness": "Uptrain-Response-Completeness",
    "UpTrain_Response_Conciseness": "Uptrain-Response-Conciseness",
    "UpTrain_Response_Relevance": "Uptrain-Response-Relevance",
    "UpTrain_Response_Valid": "Uptrain-Response-Valid",
    "UpTrain_Response_Response_Matching": "Uptrain-Response-Matching",

    "UpTrain_Retrieval_Code_Hallucination": "Uptrain-Retrieval-Code-Hallucination",
    
    # DeepEval metrics
    "DeepEval_retrieval_contextualPrecision": "DeepEval-Context-Recall",
    "DeepEval_retrieval_contextualRecall": "DeepEval-Context-Relevance",
    "DeepEval_retrieval_contextualRelevancy": "Uptrain-Context-Consistency",
    "UpTrain_Response_Consistency": "Uptrain-Context-Utilization",
    "UpTrain_Retrieval_Context_Utilization": "Uptrain-Factual-Accuracy",
    "UpTrain_Retrieval_Factual_Accuracy": "Uptrain-Factual-Accuracy",
    "DeepEval_retrieval_faithfulness": "DeepEval-Context-Faithfulness",
    "DeepEval_response_hallucination": "DeepEval-Context-Hallucination",
    
    # others not in the docs but in the code
    "DeepEval_response_bias": "DeepEval-Response-Bias",
    "DeepEval_response_toxicity": "DeepEval-Response-Toxicity",
}

# frontend values
FRONTEND_AVAILABLE_METRICS = [METRIC_DISPLAY_MAP.get(metric, metric) for metric in AVAILABLE_METRICS]
# reverse map
DISPLAY_TO_BACKEND_METRIC_MAP = {v: k for k, v in METRIC_DISPLAY_MAP.items()}

@st.cache_resource(show_spinner=False)
def get_query():
    return run(cli=False)

@st.cache_resource(show_spinner=False)
def get_qa_dataset_(dataset,files=None):
    return get_qa_dataset(dataset,files=files)

@st.cache_resource(show_spinner=False)
def get_index():
    return build_index(st.session_state.qa_dataset['documents'])

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
        
        # find the backend dataset value
        backend_dataset = cfg.dataset if cfg.dataset in DATASET_OPTIONS else DATASET_OPTIONS[0]
        # find the frontend dataset value and index
        frontend_dataset = DATASET_DISPLAY_MAP.get(backend_dataset, backend_dataset)
        frontend_index = FRONTEND_DATASET_OPTIONS.index(frontend_dataset)

        # selected_frontend_dataset = st.selectbox(
        #     "Dataset",
        #     options=FRONTEND_DATASET_OPTIONS,
        #     index=frontend_index,
        #     key="dataset"
        # )
        if 'dataset' not in st.session_state:
            st.session_state.dataset = "HotpotQA"

        cols = st.columns(3)
        st.markdown("""
        <style>
            .stCard {
                background-color: white !important;
            }
            .card-body {
                background-color: white !important;
            }
            .card {
                background-color: white !important;
            }
            /* 修改 iframe 的高度 */
            iframe.stCustomComponentV1 {
                height: 400px !important;  /* 调整这个值来改变卡片高度 */
            }
            /* 减少卡片之间的间距 */
            div[data-testid="column"] {
                margin-top: 0 !important;
                margin-bottom: 0 !important;
            }
            .block-container {
                padding-top: 2rem !important;
                padding-bottom: 2rem !important;
            }
        </style>
        """, unsafe_allow_html=True)

        for i, (dataset_name, info) in enumerate(DATASETS_INFO.items()):
            with cols[i]:
                is_selected = st.session_state.dataset == dataset_name

                features_text = "\n".join([
                    f"• {k}: {'✓' if v else '✗'}"
                    for k, v in info['features'].items()
                ])

                size_text = (
                    f"Train: {info['size']['train']}\n"
                    f"Val: {info['size']['validation']}\n"
                    f"Test: {info['size']['test']}"
                )

                clicked = card(
                    title=dataset_name,
                    text=(
                        f"📊 Dataset Size:\n{size_text}\n"
                        f"📚 Corpus:\n"
                        f"• Documents: {info['corpus']['documents']}"
                        # f"✨ Features:\n{features_text}"
                    ),
                    image="https://hotpotqa.github.io/img/home-bg.jpg",
                    styles={
                        "card": {
                            "width": "100%",
                            "height": "300px",
                            "border-radius": "10px",
                            "box-shadow": "0 0 10px rgba(0,0,0,0.1)",
                            "background-color": "#ffffff",  # 设置背景色为白色
                            "border": "4px solid #2E86C1" if is_selected else "1px solid #e0e0e0",
                        },
                        "title": {
                            "font-size": "20px",
                            "font-weight": "bold",
                            "color": "#ffffff",  # 标题改为黑色
                            "margin-bottom": "10px"
                        },
                        "text": {
                            "font-size": "14px",
                            "line-height": "1.5",
                            "white-space": "pre-line",
                            "color": "#ffffff"  # 文本改为黑色
                        }
                    }
                )

                if clicked:
                    st.session_state.dataset = dataset_name
                    st.rerun()

        selected_frontend_dataset = st.session_state.dataset

        # reverse map: from the frontend dataset to the backend dataset
        DISPLAY_TO_BACKEND_MAP = {v: k for k, v in DATASET_DISPLAY_MAP.items()}
        chosen_backend_dataset = DISPLAY_TO_BACKEND_MAP.get(selected_frontend_dataset, selected_frontend_dataset)
        cfg.dataset = chosen_backend_dataset
        # custom dataset
        st.markdown("---")
        st.markdown("## Or your own dataset")
        tab1, tab2 = st.tabs(["Upload JSON", "Generate from Folder"])
        
        with tab1:
            with st.expander("About Custom Dataset"):
                st.markdown("""
                ### Upload Custom Dataset
                
                Please upload a JSON file containing your dataset. The JSON file should be a list of objects with the following format:
                ```json
                [
                    {
                        "question": "What is the capital of France?",
                        "answer": "Paris",
                        "file_paths": "path/to/document.txt",
                        "source_text": "source of the document"
                        // or multiple files
                        // "file_paths": ["path/to/doc1.txt", "path/to/doc2.txt"]
                        // "source": ["source of the document 1", "source of the document 2"]
                    },
                    ...
                ]
                ```
                Note: 
                1. Make sure all file paths in the JSON are accessible from the server.
                2. Supported file formats: txt, md, pdf, html, json, csv, etc.
                3. Each question can reference one or multiple documents.
                4. The system will automatically process and index all documents.
                """)
            
            uploaded_file = st.file_uploader("Upload your dataset", type=["json"])
        
        with tab2:
            st.markdown("""
            ### Generate QA Pairs from Documents
            
            Upload a folder containing your documents, and the system will:
            1. Read all documents in the folder (including subfolders)
            2. Use AI to generate relevant questions and answers
            3. Create a dataset in the required format
            
            Supported file formats:
            - Text files (.txt)
            - Markdown files (.md)
            - PDF documents (.pdf)
            - HTML files (.html)
            - And more...
            
            Note:
            - If the sentence length is set to -1, the system will use file level as the unit.
            - If the sentence length is set to a positive number, the system will split the document into chunks of the specified length.
            """)
            
            folder_path = st.text_input("Enter folder path", value="./data/documents")
            num_questions = st.number_input("Number of questions per file", min_value=1, value=3)
            output_file = st.text_input("Output JSON file path", value="./data/generated_qa.json")
            sentence_length = st.number_input("Sentence length", value=-1)
            if st.button("Generate QA Dataset"):
                try:
                    with st.spinner("Generating QA pairs from documents..."):
                        from xrag.data.qa_loader import generate_qa_from_folder
                        qa_pairs = generate_qa_from_folder(folder_path, output_file, num_questions, sentence_length=sentence_length)
                        st.success(f"Successfully generated {len(qa_pairs)} QA pairs!")
                        # 自动加载生成的数据集
                        st.session_state.qa_dataset = get_qa_dataset_("custom", output_file)
                        cfg.dataset = "custom"
                        st.session_state.step = 2
                        st.rerun()
                except Exception as e:
                    st.error(f"Error generating QA dataset: {str(e)}")

        if st.button("Load Dataset"):
            if uploaded_file is not None:
                try:
                    st.session_state.step = 2
                    with st.spinner("Loading Dataset..."):
                        st.session_state.qa_dataset = get_qa_dataset_("custom", uploaded_file)
                        cfg.dataset = "custom"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
            else:
                with st.spinner("Loading Dataset..."):
                    st.session_state.qa_dataset = get_qa_dataset_(cfg.dataset)
                st.session_state.step = 2
                st.rerun()

    if st.session_state.step == 2:
        st.header("Configure your RAG Index")
        st.subheader("Settings")

        cfg.llm = st.selectbox("LLM", options=LLM_OPTIONS, index=LLM_OPTIONS.index(cfg.llm) if cfg.llm in LLM_OPTIONS else 0)
        if cfg.llm == "ollama":
            cfg.ollama_model = st.text_input("Your Ollama Model", value=cfg.ollama_model)
        elif cfg.llm == 'huggingface':
            cfg.huggingface_model = st.selectbox("HuggingFace Model", options=HF_MODEL_OPTIONS, index=HF_MODEL_OPTIONS.index(cfg.huggingface_model) if cfg.huggingface_model in HF_MODEL_OPTIONS else 0)
            cfg.auth_token = st.text_input("Your Auth Token", value=cfg.auth_token)
        elif cfg.llm == 'openai':
            cfg.api_key = st.text_input("API Key", value=cfg.api_key, type="password")
            cfg.api_base = st.text_input("API Base", value=cfg.api_base)
            cfg.api_name = st.text_input("Model Name", value=cfg.api_name)

        st.markdown("---")
        cfg.embeddings = st.selectbox("Embeddings", options=EMBEDDING_OPTIONS, index=EMBEDDING_OPTIONS.index(cfg.embeddings) if cfg.embeddings in EMBEDDING_OPTIONS else 0)
        cfg.split_type = st.selectbox("Split Type", options=SPLIT_TYPE_OPTIONS, index=SPLIT_TYPE_OPTIONS.index(cfg.split_type))
        cfg.chunk_size = st.number_input("Chunk Size", min_value=1, value=cfg.chunk_size, step=1)
        # cfg.source_dir = st.text_input("Source Directory", value=cfg.source_dir)
        cfg.persist_dir = st.text_input("Persist Directory", value=cfg.persist_dir)

        # 返回或者继续
        c1,c_,c2 = st.columns([1,4,1])
        with c1:
            if st.button("Back"):
                st.session_state.step = 1
                st.rerun()
        with c2:
            if st.button("Build Index"):
                st.session_state.step = 3
                with st.spinner("Building Index..."):
                    st.session_state.index, st.session_state.hierarchical_storage_context = get_index()
                st.rerun()
            # Evaluation Metrics Selection

    if st.session_state.step == 3:
        st.header("Configure your RAG Query Engine")
        cfg.retriever = st.selectbox("Advanced Retriever", options=RETRIEVER_OPTIONS, index=RETRIEVER_OPTIONS.index(
            cfg.retriever) if cfg.retriever in RETRIEVER_OPTIONS else 0)
        cfg.retriever_mode = st.selectbox("Retriever Mode", options=[0, 1], index=cfg.retriever_mode)
        cfg.query_transform = st.selectbox("Pre-retriever", options=QUERY_TRANSFORM_OPTIONS,
                                           index=QUERY_TRANSFORM_OPTIONS.index(
                                               cfg.query_transform) if cfg.query_transform in QUERY_TRANSFORM_OPTIONS else 0)
        cfg.postprocess_rerank = st.selectbox("Post-process", options=POSTPROCESS_RERANK_OPTIONS,
                                              index=POSTPROCESS_RERANK_OPTIONS.index(
                                                  cfg.postprocess_rerank) if cfg.postprocess_rerank in POSTPROCESS_RERANK_OPTIONS else 0)
        
        cfg.text_qa_template_str = st.text_area("Text QA Template", value=cfg.text_qa_template_str)
        cfg.refine_template_str = st.text_area("Refine Template", value=cfg.refine_template_str)

        c1, c_, c2 = st.columns([1, 4, 1])
        with c1:
            if st.button("Back"):
                st.session_state.step = 2
                st.rerun()
        with c2:
            if st.button("Build Query Engine"):
                st.session_state.step = 4
                with st.spinner("Building Query Engine..."):
                    st.session_state.query_engine = get_query_engine()
                st.rerun()
    if st.session_state.step == 4:
        st.header("Evaluate your RAG Model with single question")
        prompt = st.text_input('Input your question here')
        if st.button("Evaluate Your Question"):
            response = transform_and_query(prompt, cfg, st.session_state.query_engine)
            st.write(response.response)

            # Display source text
            with st.expander('Source Text'):
                st.write(response.get_formatted_sources(length=1024))

        c1, c_, c2 = st.columns([1, 4, 1])
        with c1:
            if st.button("Back"):
                st.session_state.step = 3
                st.rerun()

        with c2:
            if st.button("Evaluate Your Dataset"):
                st.session_state.step = 5
                st.rerun()

    if st.session_state.step == 5:
        st.header("Evaluate your RAG Model with your dataset")
        
        # 确保默认选中的 metrics 按 AVAILABLE_METRICS 的顺序排列
        sorted_default_metrics = [metric for metric in AVAILABLE_METRICS if metric in cfg.metrics]
        sorted_frontend_default_metrics = [METRIC_DISPLAY_MAP.get(metric, metric) for metric in sorted_default_metrics]
        
        # 用户在前端选择评测指标
        selected_frontend_metrics = st.multiselect(
            "Evaluation Metrics",
            options=FRONTEND_AVAILABLE_METRICS,
            default=sorted_frontend_default_metrics
        )
        
        # 将前端选择的显示名称映射回后端的真实名称
        selected_backend_metrics = [DISPLAY_TO_BACKEND_METRIC_MAP.get(metric, metric) for metric in selected_frontend_metrics]
        cfg.metrics = selected_backend_metrics

        
        # 其他输入
        cfg.test_init_total_number_documents = st.number_input(
            "Total number of documents to evaluate", 
            min_value=1, 
            value=cfg.test_init_total_number_documents, 
            step=1
        )

        c1, c_, c2 = st.columns([1, 4, 1])
        with c1:
            if st.button("Back"):
                st.session_state.step = 4
                st.rerun()

        with c2:
            start_evaluation = st.button("Start Evaluation")
        if start_evaluation:
            all_num = 0
            metrics = cfg.metrics.copy()
            evaluateResults = EvaluationResult(metrics=metrics)
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
                eval_result = evaluating(
                    question, response, actual_response, retrieval_context, retrieval_ids,
                    expected_answer, golden_context, golden_context_ids, evaluateResults.metrics,
                    evalAgent
                )
                with st.expander(question):
                    st.markdown("### Answer")
                    st.markdown(response.response)
                    st.markdown('### Retrieval context')
                    st.markdown('\n\n'.join(retrieval_context))
                    st.markdown('### Expected answer')
                    st.markdown(expected_answer)
                    st.markdown('### Golden context')
                    st.markdown('\n\n'.join(golden_context))

                print(eval_result)

                evaluateResults.add(eval_result)
                all_num += 1
                st.markdown(evaluateResults.get_results_str())

            st.success("Evaluation complete!")
            st.session_state.evaluation_results = evaluateResults

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
