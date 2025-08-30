import os
import toml
import shutil
import pkg_resources
from .utils import get_module_logger

logger = get_module_logger(__name__)


def create_default_config(config_file_path):
    """Create a default config file if it doesn't exist."""
    try:
        # Get the default config file from the package
        default_config = pkg_resources.resource_filename('xrag', 'default_config.toml')
        # Copy it to the target location
        shutil.copy2(default_config, config_file_path)
    except Exception:
        # If package resource not found, create a new config file with default values
        default_config = {
            "api_keys": {
                "api_key": "sk-xxxx",
                "api_base": "https://api.openai.com/v1",
                "api_name": "gpt-4",
                "auth_token": "hf_xxx"
            },
            "logging": {
                "log_level": "INFO",
                "log_file": "logs/xrag.log",
                "log_format": "[%(asctime)s,%(msecs)03d] %(levelname)-8s - (%(name)s) %(filename)s, ln %(lineno)d: %(message)s",
            },
            "settings": {
                "llm": "openai",
                "ollama_model": "llama2:7b",
                "huggingface_model": "llama",
                "embeddings": "BAAI/bge-large-en-v1.5",
                "split_type": "sentence",
                "chunk_size": 128,
                "window_size": 3,
                "dataset": "hotpot_qa",
                "dataset_download_timeout": 300,
                "persist_dir": "storage",
                "llamaIndexEvaluateModel": "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
                "deepEvalEvaluateModel": "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
                "upTrainEvaluateModel": "qwen:7b-chat-v1.5-q8_0",
                "evaluateApiName": "",
                "evaluateApiKey": "",
                "evaluateApiBase": "",
                "output": "",
                "n": 100,
                "test_init_total_number_documents": 20,
                "extra_number_documents": 20,
                "extra_rate_documents": 0.1,
                "test_all_number_documents": 40,
                "experiment_1": False,
                "retriever": "BM25",
                "retriever_mode": 0,
                "postprocess_rerank": "long_context_reorder",
                "query_transform": "none",
                "metrics": ["NLG_chrf", "NLG_meteor", "NLG_wer", "NLG_cer", "NLG_chrf_pp",
                          "NLG_perplexity", "NLG_rouge_rouge1", "NLG_rouge_rouge2", 
                          "NLG_rouge_rougeL", "NLG_rouge_rougeLsum"]
            },
            "seper": {
                "enabled": False,
                "generation_model": "meta-llama/Llama-2-7b-chat-hf",
                "entailment_model": "microsoft/deberta-v2-xlarge-mnli",
                "device": "cuda",
                "num_generations": 10,
                "temperature": 1.0,
                "max_new_tokens": 128,
                "computation_chunk_size": 8,
                "prompt_type": "default",
                "max_context_words": 512,
                "use_soft_clustering": True
            },
            "prompt": {
                "text_qa_template_path": "src/xrag/prompts/text_qa_template.txt",
                "refine_template_path": "src/xrag/prompts/refine_template.txt"
            },
            "self_rag": {
                "enabled": False,
                "model_name": "selfrag/selfrag_llama2_7b",
                "retriever_model": "facebook/contriever-msmarco",
                "download_dir": "",
                "dtype": "half",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 100,
                "skip_special_tokens": False
            }
        }
        with open(config_file_path, 'w', encoding='utf-8') as f:
            toml.dump(default_config, f)


class Config:
    _instance = None  # Singleton instance

    def __new__(cls, config_file_path=None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            if config_file_path is None:
                config_file_path = 'config.toml'
            
            # Create default config if file doesn't exist
            if not os.path.exists(config_file_path):
                create_default_config(config_file_path)
            
            cls._instance.config = toml.load(config_file_path)  # Load the config only once

            # Dynamically set attributes based on TOML config
            for section, values in cls._instance.config.items():
                for key, value in values.items():
                    setattr(cls._instance, key, value)

        return cls._instance

    def _load_prompt_template(self, file_path: str):
        """Load prompt template from file."""
        try:
            # Handle relative paths from the project root
            if not os.path.isabs(file_path):
                project_root = os.path.dirname(os.path.abspath("config.toml"))
                file_path = os.path.join(project_root, file_path)

            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"Prompt template file not found: {file_path}")
            return ""
        except Exception as e:
            logger.error(f"Error loading prompt template from {file_path}: {e}")
            return ""

    @property
    def text_qa_template_str(self):
        """Get the text QA template string from file."""
        if hasattr(self, "text_qa_template_path"):
            return self._load_prompt_template(self.text_qa_template_path)
        return ""

    @property
    def refine_template_str(self):
        """Get the refine template string from file."""
        if hasattr(self, "refine_template_path"):
            return self._load_prompt_template(self.refine_template_path)
        return ""

    def update_config(self, overrides):
        for key, value in overrides.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                new_value = self._convert_type(value, type(old_value))
                setattr(self, key, new_value)
            else:
                logger.warning(f"Invalid configuration key: {key}")

    def _convert_type(self, value, to_type):
        try:
            if to_type == bool:
                return value.lower() in ('true', '1', 'yes')
            elif to_type == int:
                return int(value)
            elif to_type == float:
                return float(value)
            elif to_type == str:
                return value
            else:
                return value  # For other types
        except ValueError:
            logger.error(f"Could not convert value '{value}' to type {to_type.__name__}")
            return value


class GlobalVar:
    query_number = 0

    @staticmethod
    def set_query_number(num):
        GlobalVar.query_number = num

    @staticmethod
    def get_query_number():
        return GlobalVar.query_number


if __name__ == '__main__':
    # Usage: Create a global instance
    cfg = Config()

    # Now you can access config values directly as attributes:
    logger.info(cfg.test_init_total_number_documents)  # Outputs: 20
    logger.info(cfg.api_key)  # Outputs the API key from the toml file
