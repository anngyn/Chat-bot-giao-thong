"""
Pytest configuration and fixtures for AWS Bedrock RAG Chatbot tests
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch
from typing import Dict, Any

# Set test environment
os.environ['ENVIRONMENT'] = 'test'
os.environ['AWS_REGION'] = 'us-east-1'
os.environ['S3_BUCKET_NAME'] = 'test-rag-traffic-vi'

@pytest.fixture
def mock_aws_clients():
    """Mock AWS clients for testing"""
    with patch('shared.utils.aws_clients.boto3') as mock_boto3:
        mock_s3 = Mock()
        mock_bedrock = Mock()
        mock_eventbridge = Mock()
        mock_cloudwatch = Mock()
        
        mock_boto3.client.side_effect = lambda service, **kwargs: {
            's3': mock_s3,
            'bedrock-runtime': mock_bedrock,
            'events': mock_eventbridge,
            'cloudwatch': mock_cloudwatch
        }.get(service, Mock())
        
        yield {
            's3': mock_s3,
            'bedrock': mock_bedrock,
            'eventbridge': mock_eventbridge,
            'cloudwatch': mock_cloudwatch
        }

@pytest.fixture
def sample_text():
    """Sample Vietnamese text for testing"""
    return """
    Nghị định 100/2019/NĐ-CP quy định về xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ và đường sắt.
    
    Điều 5. Xử phạt vi phạm quy tắc giao thông đường bộ
    1. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe ô tô vi phạm một trong các lỗi sau đây:
    a) Vượt đèn đỏ, đèn vàng;
    b) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông tại nơi đường giao nhau.
    """

@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return "Mức phạt vượt đèn đỏ là bao nhiêu?"

@pytest.fixture
def temp_directory():
    """Temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_faiss():
    """Mock FAISS for testing"""
    with patch('shared.utils.vector_operations.faiss') as mock_faiss:
        mock_index = Mock()
        mock_index.ntotal = 0
        mock_index.d = 1024
        mock_index.is_trained = True
        mock_index.search.return_value = (
            [[0.1, 0.2, 0.3]],  # distances
            [[0, 1, 2]]         # indices
        )
        
        mock_faiss.IndexFlatL2.return_value = mock_index
        mock_faiss.IndexIVFFlat.return_value = mock_index
        mock_faiss.read_index.return_value = mock_index
        
        yield mock_faiss

@pytest.fixture
def lambda_context():
    """Mock Lambda context for testing"""
    context = Mock()
    context.function_name = 'test-function'
    context.function_version = '1'
    context.invoked_function_arn = 'arn:aws:lambda:us-east-1:123456789012:function:test-function'
    context.memory_limit_in_mb = 128
    context.get_remaining_time_in_millis.return_value = 30000
    return context

@pytest.fixture
def api_gateway_event():
    """Sample API Gateway event for testing"""
    return {
        'httpMethod': 'POST',
        'path': '/chat',
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': '{"query": "Mức phạt vượt đèn đỏ là bao nhiêu?"}',
        'requestContext': {
            'requestId': 'test-request-id',
            'identity': {
                'sourceIp': '127.0.0.1'
            }
        }
    }

@pytest.fixture
def s3_event():
    """Sample S3 event for testing"""
    return {
        'Records': [
            {
                'eventSource': 'aws:s3',
                'eventName': 'ObjectCreated:Put',
                's3': {
                    'bucket': {
                        'name': 'test-rag-traffic-vi'
                    },
                    'object': {
                        'key': 'raw/pdf/test-document.pdf'
                    }
                }
            }
        ]
    }