#!/bin/bash

# Ensure script stops on first error
set -e

echo "Starting local Kubernetes deployment..."

# Check if minikube is running
if ! minikube status > /dev/null 2>&1; then
    echo "Starting Minikube..."
    minikube start
fi

# Set docker to use minikube's docker daemon
eval $(minikube docker-env)

# Build the Docker image
echo "Building Docker image..."
docker build -t ml-model-api:latest -f deployment/Dockerfile .


# Deploy using Helm
echo "Deploying with Helm..."
helm upgrade --install ml-model-api deployment/helm/ml-model \
    --namespace default \
    --create-namespace \
    --wait

# Get the URL
echo "Getting service URL..."
minikube service ml-model-api-service --url

echo "Deployment complete!" 