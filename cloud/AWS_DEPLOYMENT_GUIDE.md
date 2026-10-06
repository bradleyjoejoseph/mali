# Mali — AWS Cloud Architecture & Deployment Guide

This guide details how **Mali** integrates with Amazon Web Services (AWS) for production MLOps:
1. **Artifact Storage & Versioning** via **Amazon S3**
2. **Container Registry** via **Amazon ECR**
3. **Serverless / Containerized Live Serving** via **AWS App Runner** or **ECS Fargate**

---

## 1. Amazon S3 Model Registry

Rather than committing large model weights (`.pt`) to Git, Mali automatically versions checkpoints in Amazon S3.

### S3 Architecture
```
s3://mali-model-artifacts/
├── vocab/
│   └── mali_vocab.json
└── checkpoints/
    ├── mali_v1_best.pt
    └── mali_v2_best.pt
```

### Setup S3 Bucket via AWS CLI
```bash
# 1. Create S3 Bucket (choose your region, e.g. eu-west-2 for London)
aws s3 mb s3://mali-model-artifacts --region eu-west-2

# 2. Upload Best Checkpoint manually or via Python
python -m cloud.s3_sync --action upload --local checkpoints/mali_best.pt --bucket mali-model-artifacts --key checkpoints/mali_v1.pt
python -m cloud.s3_sync --action upload --local checkpoints/mali_vocab.json --bucket mali-model-artifacts --key vocab/mali_vocab.json

# 3. Pull on any cloud machine or server:
python -m cloud.s3_sync --action download --bucket mali-model-artifacts --key checkpoints/mali_v1.pt --local checkpoints/mali_best.pt
```

---

## 2. Containerization & Amazon ECR (Elastic Container Registry)

Mali includes a production `Dockerfile` wrapping the FastAPI inference engine.

### Build and Push to Amazon ECR:
```bash
# 1. Authenticate Docker with Amazon ECR
aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.eu-west-2.amazonaws.com

# 2. Create ECR repository
aws ecr create-repository --repository-name mali-inference --region eu-west-2

# 3. Build & Tag Docker image
docker build -t mali-inference -f api/Dockerfile .
docker tag mali-inference:latest <AWS_ACCOUNT_ID>.dkr.ecr.eu-west-2.amazonaws.com/mali-inference:latest

# 4. Push to AWS
docker push <AWS_ACCOUNT_ID>.dkr.ecr.eu-west-2.amazonaws.com/mali-inference:latest
```

---

## 3. Live Deployment via AWS App Runner (Zero-Ops & Low Cost)

AWS App Runner is the ideal AWS service for ML inference because:
- It requires **zero VPC or server maintenance**.
- It provides an **automatic public HTTPS URL** with SSL.
- It auto-scales based on incoming HTTP traffic.

### Deploying via AWS Console or CLI:
1. Open the **AWS App Runner Console** $\rightarrow$ Click **Create Service**.
2. Select **Container Registry** $\rightarrow$ Choose your Amazon ECR image `mali-inference:latest`.
3. Set CPU to **1 vCPU** and Memory to **2 GB**.
4. Set Port to **8000**.
5. Click **Deploy**.

Within 3 minutes, AWS assigns your live public domain:
`https://<service-hash>.eu-west-2.awsapprunner.com`

You can test it publicly:
```bash
curl -X POST "https://<service-hash>.eu-west-2.awsapprunner.com/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "The kingdom of", "max_tokens": 100, "temperature": 0.8}'
```
Interactive Swagger documentation is available live at `https://<service-hash>.eu-west-2.awsapprunner.com/docs`.
