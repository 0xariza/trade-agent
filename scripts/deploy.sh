#!/bin/bash
# Production deployment script

echo "🚀 Deploying Alpha Arena..."

# Pull latest code
git pull origin main

# Rebuild containers
docker-compose -f docker-compose.prod.yml build

# Stop old containers
docker-compose -f docker-compose.prod.yml down

# Start new containers
docker-compose -f docker-compose.prod.yml up -d

# Run migrations
docker-compose -f docker-compose.prod.yml exec api alembic upgrade head

# Check health
sleep 10
docker-compose -f docker-compose.prod.yml ps

echo "✅ Deployment complete!"
