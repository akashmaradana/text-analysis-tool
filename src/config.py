from typing import Dict, Any

# Model configurations
BART_MODEL_NAME = "facebook/bart-large-cnn"
BERT_MODEL_NAME = "bert-base-uncased"

# Summarization parameters
DEFAULT_MAX_LENGTH = 130
DEFAULT_MIN_LENGTH = 30
DEFAULT_LENGTH_PENALTY = 2.0
DEFAULT_NUM_BEAMS = 4
DEFAULT_EARLY_STOPPING = True

# Text processing
MAX_INPUT_LENGTH = 1024
MIN_INPUT_LENGTH = 100

# API settings
MODEL_TIMEOUT = 30  # seconds

def get_bart_config() -> Dict[str, Any]:
    return {
        "max_length": DEFAULT_MAX_LENGTH,
        "min_length": DEFAULT_MIN_LENGTH,
        "length_penalty": DEFAULT_LENGTH_PENALTY,
        "num_beams": DEFAULT_NUM_BEAMS,
        "early_stopping": DEFAULT_EARLY_STOPPING
    }

def get_bert_config() -> Dict[str, Any]:
    return {
        "max_length": DEFAULT_MAX_LENGTH,
        "min_length": DEFAULT_MIN_LENGTH
    } 