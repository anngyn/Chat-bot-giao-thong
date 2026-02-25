# AWS Bedrock RAG Chatbot - Project Structure

## Overview
This project implements a serverless RAG (Retrieval-Augmented Generation) chatbot for Vietnamese traffic law using AWS Bedrock, Lambda, S3, and API Gateway.

## Directory Structure

```
bedrock-rag-chatbot/
├── lambda_functions/              # AWS Lambda function implementations
│   ├── rag_orchestrator/         # Main RAG processing Lambda
│   │   ├── lambda_function.py    # Main handler
│   │   ├── requirements.txt      # Lambda-specific dependencies
│   │   └── __init__.py
│   ├── document_indexer/         # Document processing Lambda
│   │   ├── lambda_function.py    # Main handler
│   │   ├── requirements.txt      # Lambda-specific dependencies
│   │   └── __init__.py
│   └── health_check/             # System health monitoring Lambda
│       ├── lambda_function.py    # Main handler
│       ├── requirements.txt      # Lambda-specific dependencies
│       └── __init__.py
├── infrastructure/               # Infrastructure as Code (CloudFormation)
│   ├── cloudformation-template.yaml  # Main CloudFormation template
│   └── __init__.py
├── shared/                      # Shared utilities and libraries
│   ├── config/                  # Configuration management
│   │   ├── settings.py          # Application configuration
│   │   ├── constants.py         # System constants
│   │   └── __init__.py
│   ├── models/                  # Data models and schemas
│   │   ├── data_models.py       # Core data structures
│   │   └── __init__.py
│   ├── utils/                   # Common utilities
│   │   ├── aws_clients.py       # AWS service clients
│   │   ├── error_handling.py    # Error handling utilities
│   │   ├── logging_utils.py     # Structured logging
│   │   ├── text_processing.py   # Vietnamese text processing
│   │   ├── vector_operations.py # FAISS vector operations
│   │   └── __init__.py
│   └── __init__.py
├── tests/                       # Test suite
│   ├── test_shared/             # Tests for shared utilities
│   │   ├── test_config.py       # Configuration tests
│   │   └── __init__.py
│   ├── test_lambda/             # Tests for Lambda functions
│   │   └── __init__.py
│   ├── conftest.py              # Pytest configuration
│   └── __init__.py
├── data/                        # Data files and documents
│   ├── vietnamese-stopwords-dash.txt  # Vietnamese stopwords
│   └── [other data files]
├── .kiro/specs/bedrock-rag-chatbot/  # Project specifications
│   ├── requirements.md          # Feature requirements
│   ├── design.md               # System design
│   └── tasks.md                # Implementation tasks
├── bedrock_requirements.txt     # Python dependencies for Bedrock RAG
├── requirements-dev.txt         # Development dependencies
├── .env.template               # Environment configuration template
├── deployment.yaml             # Deployment configuration
├── Makefile                    # Development automation
├── setup.py                    # Project setup script
└── PROJECT_STRUCTURE.md       # This file
```

## Key Components

### Lambda Functions
- **RAG Orchestrator**: Handles chat queries, vector search, and LLM inference
- **Document Indexer**: Processes uploaded documents and updates vector store
- **Health Check**: Monitors system status and connectivity

### Shared Libraries
- **Config**: Environment configuration and constants management
- **Models**: Data models for documents, queries, and responses
- **Utils**: Common utilities for text processing, vector operations, etc.

### Infrastructure
- CloudFormation templates for AWS resource provisioning
- IAM roles and policies with least privilege access
- S3 bucket configuration with proper security settings

## Configuration

### Environment Variables
Copy `.env.template` to `.env` and configure:
- AWS credentials and region
- S3 bucket configuration
- Bedrock model IDs
- Vector search parameters
- Text processing settings

### Dependencies
Install dependencies from `bedrock_requirements.txt`:
```bash
pip install -r bedrock_requirements.txt
```

## AWS Services Used

### Free Tier Services
- **S3**: Document storage and vector index storage (<5GB)
- **Lambda**: Serverless compute (<1M requests/month)
- **API Gateway**: REST API endpoints (<1M requests/month)
- **CloudWatch**: Logging and monitoring (<5GB logs/month)
- **EventBridge**: Event-driven document processing

### Bedrock Models
- **Embeddings**: amazon.titan-embed-text-v2:0
- **LLM**: meta.llama3-8b-instruct-v1:0
- **Fallback LLM**: mistral.mistral-small-v1:0

## Development Workflow

1. **Setup**: Configure environment variables and install dependencies
2. **Development**: Implement Lambda functions and shared utilities
3. **Testing**: Unit tests and integration tests
4. **Deployment**: Use CloudFormation for infrastructure deployment
5. **Monitoring**: CloudWatch metrics and logging

## Security Considerations

- IAM roles with least privilege access
- S3 buckets with private access only
- PII masking in logs
- Input validation and sanitization
- Rate limiting and guardrails

## Vietnamese Language Support

- Vietnamese text segmentation with PyVi
- Vietnamese stopwords filtering
- Unicode normalization
- Accent preservation for accurate search

## Next Steps

1. Implement core utilities and shared libraries (Task 3)
2. Set up infrastructure with CloudFormation (Task 2)
3. Develop Lambda functions (Tasks 4-6)
4. Configure monitoring and logging (Task 8)
5. Deploy and test the complete system