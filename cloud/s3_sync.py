import os
import argparse
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def get_s3_client():
    """Initialize boto3 S3 client with environment or AWS CLI credentials."""
    try:
        return boto3.client("s3")
    except Exception as e:
        print(f"[AWS Warning] Unable to initialize S3 client: {e}")
        return None

def upload_to_s3(local_file: str, bucket: str, s3_key: str) -> bool:
    """Upload a model checkpoint or vocabulary artifact to Amazon S3."""
    if not os.path.exists(local_file):
        print(f"[AWS Error] Local file does not exist: {local_file}")
        return False
    
    s3 = get_s3_client()
    if s3 is None:
        return False

    try:
        print(f"[AWS S3] Uploading {local_file} -> s3://{bucket}/{s3_key}...")
        s3.upload_file(local_file, bucket, s3_key)
        print(f"[AWS S3] ✅ Successfully uploaded to s3://{bucket}/{s3_key}")
        return True
    except NoCredentialsError:
        print("[AWS Error] AWS credentials not found. Run 'aws configure' or set AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY.")
        return False
    except ClientError as e:
        print(f"[AWS Error] Client error during upload: {e}")
        return False

def download_from_s3(bucket: str, s3_key: str, local_file: str) -> bool:
    """Download a model checkpoint or vocabulary artifact from Amazon S3."""
    s3 = get_s3_client()
    if s3 is None:
        return False

    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    try:
        print(f"[AWS S3] Downloading s3://{bucket}/{s3_key} -> {local_file}...")
        s3.download_file(bucket, s3_key, local_file)
        print(f"[AWS S3] ✅ Download complete: {local_file}")
        return True
    except Exception as e:
        print(f"[AWS Error] Download failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Mali AWS S3 Artifact Manager")
    parser.add_argument("--action", choices=["upload", "download"], required=True)
    parser.add_argument("--local", type=str, required=True, help="Local file path")
    parser.add_argument("--bucket", type=str, default="mali-model-artifacts", help="S3 bucket name")
    parser.add_argument("--key", type=str, required=True, help="S3 object key")
    args = parser.parse_args()

    if args.action == "upload":
        upload_to_s3(args.local, args.bucket, args.key)
    else:
        download_from_s3(args.bucket, args.key, args.local)

if __name__ == "__main__":
    main()
