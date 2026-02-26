#!/usr/bin/env python3
"""
Simple script to make a test Bedrock API call and verify it appears in CloudTrail.
"""

import boto3
import json
from datetime import datetime

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

print("Making a test Bedrock API call...")
print("="*60)

try:
    # Make a simple API call to Claude
    response = bedrock.invoke_model(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": "Say 'Hello from Open-CITE test'"
                }
            ]
        })
    )
    
    # Parse response
    response_body = json.loads(response['body'].read())
    print("✅ Bedrock API call successful!")
    print(f"Response: {response_body.get('content', [{}])[0].get('text', 'N/A')}")
    print(f"\nTimestamp: {datetime.utcnow().isoformat()}Z")
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Wait 5-15 minutes for CloudTrail to log this event")
    print("2. Run: python verify_bedrock.py")
    print("3. You should see this invocation appear!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nCommon issues:")
    print("  - Model not available in your region")
    print("  - Model access not granted (check Bedrock model access in AWS Console)")
    print("  - Wrong region")
    print("\nTo grant model access:")
    print("  1. AWS Console → Bedrock → Model access")
    print("  2. Request access to Claude models")
    print("  3. Wait for approval (usually instant)")