"""
AWS discovery plugins for OpenCITE.

This package provides discovery plugins for AWS AI/ML services:
- AWS Bedrock: Foundation models, custom models, invocations
- AWS SageMaker: Endpoints, models, training jobs
"""

from open_cite.plugins.aws.bedrock import AWSBedrockPlugin
from open_cite.plugins.aws.sagemaker import AWSSageMakerPlugin

__all__ = ["AWSBedrockPlugin", "AWSSageMakerPlugin"]
