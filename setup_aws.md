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

