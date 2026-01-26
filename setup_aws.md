# AWS CLI Setup for OpenCITE

## Step 1: Install AWS CLI

### Option A: Using MSI Installer (Easiest)

1. Download the AWS CLI MSI installer:
   - Go to: https://awscli.amazonaws.com/AWSCLIV2.msi
   - Or search "AWS CLI Windows MSI" in your browser

2. Run the installer and follow the prompts

3. Verify installation:
   ```powershell
   aws --version
   ```

### Option B: Using pip (Alternative)

```powershell
pip install awscli
```

## Step 2: Configure AWS Credentials

After installation, configure your root credentials:

```powershell
aws configure
```

You'll be prompted for:
- **AWS Access Key ID**: Your root account access key
- **AWS Secret Access Key**: Your root account secret key
- **Default region name**: e.g., `us-east-1` (Bedrock is available in us-east-1, us-west-2)
- **Default output format**: `json` (recommended)

This creates:
- `~/.aws/credentials` - Your credentials
- `~/.aws/config` - Your region and output settings

## Step 3: Verify Credentials

```powershell
aws sts get-caller-identity
```

Should return your AWS account ID and user ARN.

## Step 4: Test Bedrock Access

```powershell
# List available Bedrock foundation models
aws bedrock list-foundation-models --region us-east-1

# Or check if Bedrock is available in your region
aws bedrock list-foundation-models --region us-west-2
```

## Step 5: Use with OpenCITE

Once AWS CLI is configured, OpenCITE will automatically use those credentials:

```python
from open_cite import OpenCiteClient

# Automatically uses credentials from ~/.aws/credentials
client = OpenCiteClient(
    enable_aws_bedrock=True,
    aws_region="us-east-1"  # or us-west-2
)

# Verify connection
status = client.verify_bedrock_connection()
print(status)

# List available models
models = client.list_bedrock_models()
for model in models[:5]:  # Show first 5
    print(f"{model['name']} ({model['provider']})")
```

## Step 6: Test Bedrock Model Invocations

⚠️ **Important**: OpenCITE tracks **direct Bedrock model invocations** (API calls to `bedrock-runtime:InvokeModel`), not Bedrock Agent invocations. Console-based Bedrock Agents use different APIs and won't appear in these tests.

### Option A: Make a Test Bedrock Call

Use the provided script to make a real Bedrock API call that will appear in CloudTrail:

```powershell
python test_bedrock_call.py
```

This script:
- Makes a direct `InvokeModel` API call to Claude
- Generates a CloudTrail event that OpenCITE can detect
- Shows you the response and timestamp

**Note**: CloudTrail events can take 5-15 minutes to appear. After running this script, wait a few minutes before verifying.

### Option B: Verify Bedrock Setup and Activity

Use the verification script to check your Bedrock connection and see recent model invocations:

```powershell
python verify_bedrock.py
```

This script will:
1. ✅ Verify your Bedrock connection
2. 📋 List available foundation models
3. 🔍 Check CloudTrail for recent model invocations (last 7 days)
4. 📊 Show usage statistics by model

**What it shows**:
- Direct model API calls (`bedrock-runtime:InvokeModel`)
- Model usage statistics
- Recent activity from your code-based Bedrock usage

**What it doesn't show**:
- Bedrock Agent invocations (console-based agents use `InvokeAgent` API)
- Agent orchestration events

### Understanding the Difference

- **Model Invocations** (tracked by OpenCITE):
  - Direct API calls: `bedrock_runtime.invoke_model()`
  - Code-based usage (Python, Node.js, etc.)
  - Shows up in CloudTrail as `InvokeModel` events
  - What most developers use in production

- **Agent Invocations** (not tracked by OpenCITE):
  - Console-based Bedrock Agents
  - Uses `InvokeAgent` API
  - Different CloudTrail event type
  - Higher-level abstraction for non-developers

If you're testing with console agents, you won't see activity in `verify_bedrock.py`. Use `test_bedrock_call.py` to generate a real model invocation that OpenCITE can track.

## Important Notes

⚠️ **Root Credentials Warning**: Using root account credentials is not recommended for security. Consider:
1. Creating an IAM user with limited permissions
2. Using IAM roles for applications
3. Rotating credentials regularly

### Minimum IAM Permissions for Bedrock

If you create an IAM user instead of using root, they need:
- `bedrock:ListFoundationModels`
- `bedrock:GetFoundationModel`
- `bedrock:ListCustomModels`
- `sts:GetCallerIdentity`

## Troubleshooting

### "Unable to locate credentials"
- Make sure you ran `aws configure`
- Check `~/.aws/credentials` exists

### "Access Denied"
- Verify your credentials have Bedrock permissions
- Check the region (Bedrock not available in all regions)

### "Service not available in region"
- Try `us-east-1` or `us-west-2` (Bedrock is available there)

