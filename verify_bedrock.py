#!/usr/bin/env python3
"""
Quick script to verify Bedrock connection and see your test agent activity.
"""

from open_cite import OpenCiteClient
import json

# Initialize client
print("Connecting to AWS Bedrock...")
client = OpenCiteClient(
    enable_aws_bedrock=True,
    aws_region="us-east-1"  # Change if your region is different
)

# 1. Verify connection
print("\n" + "="*60)
print("1. Verifying Connection")
print("="*60)
status = client.verify_bedrock_connection()
if status["success"]:
    print(f"✅ Connected successfully!")
    print(f"   Region: {status['region']}")
    print(f"   Account ID: {status.get('account_id', 'N/A')}")
    print(f"   Foundation models available: {status.get('foundation_models_available', 0)}")
else:
    print(f"❌ Connection failed: {status.get('error', 'Unknown error')}")
    exit(1)

# 2. List available models
print("\n" + "="*60)
print("2. Available Foundation Models")
print("="*60)
models = client.list_bedrock_models()
print(f"Found {len(models)} models")
if models:
    print("\nSample models:")
    for model in models[:5]:
        print(f"  - {model['name']} ({model['provider']})")

# 3. Check for recent activity (invocations)
print("\n" + "="*60)
print("3. Recent Activity (Last 7 Days)")
print("="*60)
print("Checking CloudTrail for Bedrock invocations...")
invocations = client.list_bedrock_invocations(days=7)

if invocations:
    print(f"✅ Found {len(invocations)} invocations!")
    print("\nRecent activity:")
    for inv in invocations[:10]:  # Show first 10
        print(f"\n  {inv.get('timestamp', 'N/A')}")
        print(f"    Model: {inv.get('model_id', 'N/A')}")
        print(f"    User: {inv.get('user_arn', 'N/A')}")
        if inv.get('metadata', {}).get('error_code'):
            print(f"    ⚠️  Error: {inv['metadata']['error_code']}")
else:
    print("⚠️  No invocations found")
    print("\nThis could mean:")
    print("  - Your test agent hasn't made any Bedrock calls yet")
    print("  - CloudTrail is not enabled")
    print("  - CloudTrail events haven't appeared yet (can take 5-15 minutes)")

# 4. Get usage statistics
print("\n" + "="*60)
print("4. Usage Statistics by Model")
print("="*60)
usage = client.get_bedrock_usage_by_model(days=7)

if usage:
    print(f"✅ Found usage for {len(usage)} models:")
    # Sort by invocation count
    sorted_usage = sorted(usage.items(), key=lambda x: x[1]['invocation_count'], reverse=True)
    for model_id, stats in sorted_usage:
        print(f"\n  {model_id}:")
        print(f"    Invocations: {stats['invocation_count']}")
        print(f"    Unique users: {stats['unique_user_count']}")
        print(f"    Last used: {stats.get('last_seen', 'N/A')}")
else:
    print("⚠️  No usage statistics available")
    print("   (No Bedrock invocations found in CloudTrail)")

# 5. Summary
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Connection: {'✅ Working' if status['success'] else '❌ Failed'}")
print(f"Models available: {len(models)}")
print(f"Recent invocations: {len(invocations)}")
print(f"Active models: {len(usage)}")

if invocations:
    print("\n✅ Your test agent activity is visible!")
else:
    print("\n💡 To see your test agent activity:")
    print("   1. Make sure CloudTrail is enabled")
    print("   2. Run your test agent to make Bedrock API calls")
    print("   3. Wait 5-15 minutes for CloudTrail events to appear")
    print("   4. Run this script again")

