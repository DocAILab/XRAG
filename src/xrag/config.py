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
                "dataset": "hotpot_qa",
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
            "prompt": {
                "text_qa_template_str": "Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the question: {query_str}\n",
                "refine_template_str": "We have the opportunity to refine the original answer(only if needed) with some more context below.\n------------\n{context_msg}\n------------\nGiven the new context, refine the original answer to better answer the question: {query_str}. If the context isn't useful, output the original answer again.\nOriginal Answer: {existing_answer}"
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
