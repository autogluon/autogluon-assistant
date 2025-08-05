# AWS Credential Refresh Tool

This tool automatically refreshes AWS temporary credentials using STS assume-role and saves them to a file that can be used by the SageMaker LLM provider.

## Prerequisites

- AWS CLI installed and configured
- `jq` command-line tool installed
- IAM permissions to assume the target role

## Usage

```bash
./refresh_credentials.sh [options]
```

### Options

- `-r, --role-arn ARN` (Required): AWS IAM Role ARN to assume
- `-o, --output-file FILE` (Required): Path to save the credentials
- `-i, --interval SECONDS` (Optional): Refresh interval in seconds (default: 1800, 30 minutes)
- `-s, --session-name NAME` (Optional): Role session name (default: "autogluon-session")
- `-d, --duration SECONDS` (Optional): Session duration in seconds (default: 3600, 1 hour)
- `-n, --no-loop` (Optional): Run once, don't loop continuously
- `-h, --help`: Show usage information

## Examples

### Run once and exit

```bash
./refresh_credentials.sh \
  --role-arn arn:aws:iam::123456789012:role/ExampleRole \
  --output-file /path/to/credentials.txt \
  --no-loop
```

### Run continuously with custom settings

```bash
./refresh_credentials.sh \
  --role-arn arn:aws:iam::123456789012:role/ExampleRole \
  --output-file /path/to/credentials.txt \
  --interval 900 \
  --session-name "my-custom-session" \
  --duration 7200
```

## Credentials File Format

The script generates a JSON file with the following structure:

```json
{
  "Credentials": {
    "AccessKeyId": "ASIA...",
    "SecretAccessKey": "...",
    "SessionToken": "...",
    "Expiration": "2023-01-01T00:00:00Z"
  }
}
```

This file is compatible with the SageMaker LLM provider's credential refresh mechanism.

## Using with AutoGluon Assistant

In your SageMaker configuration, specify the path to the credentials file:

```yaml
llm:
  provider: sagemaker
  endpoint_name: your-endpoint-name
  region_name: us-west-2
  creds_file: /path/to/credentials.txt
```

## Security Notes

- The credentials file is created with permissions restricted to the current user (mode 600)
- Always use the minimum required permissions when creating IAM roles
- Consider running this process in a secure environment