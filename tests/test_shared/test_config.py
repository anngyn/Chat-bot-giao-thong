"""
Tests for configuration management
"""

import pytest
import os
from shared.config.settings import AppConfig, get_config
from shared.config.constants import BEDROCK_MODELS, VECTOR_CONFIG

def test_app_config_creation():
    """Test AppConfig creation from environment"""
    config = AppConfig.from_env()
    
    assert config.aws.region == 'us-east-1'
    assert config.s3.bucket_name == 'test-rag-traffic-vi'
    assert config.bedrock.embedding_model_id == BEDROCK_MODELS['embedding']['primary']
    assert config.vector.embedding_dimension == VECTOR_CONFIG['embedding_dimensions']['titan-v2']

def test_get_config():
    """Test global config getter"""
    config = get_config()
    assert isinstance(config, AppConfig)
    assert config.environment == 'test'

def test_config_to_dict():
    """Test config serialization"""
    config = AppConfig.from_env()
    config_dict = config.to_dict()
    
    assert 'aws' in config_dict
    assert 'bedrock' in config_dict
    assert 'vector' in config_dict
    assert config_dict['aws']['region'] == 'us-east-1'

def test_bedrock_models_constants():
    """Test Bedrock model constants"""
    assert 'embedding' in BEDROCK_MODELS
    assert 'llm' in BEDROCK_MODELS
    assert 'primary' in BEDROCK_MODELS['embedding']
    assert 'primary' in BEDROCK_MODELS['llm']

def test_vector_config_constants():
    """Test vector configuration constants"""
    assert 'faiss_index_type' in VECTOR_CONFIG
    assert 'embedding_dimensions' in VECTOR_CONFIG
    assert 'titan-v2' in VECTOR_CONFIG['embedding_dimensions']
    assert VECTOR_CONFIG['embedding_dimensions']['titan-v2'] == 1024