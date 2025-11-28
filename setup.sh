#!/bin/bash

# Alpha Arena - Complete Setup Script with All Files
# This script creates the project structure and all necessary files

set -e  # Exit on error

echo "🚀 Alpha Arena - Complete Project Setup"
echo "========================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_NAME="alpha-arena"

echo -e "${GREEN}Creating project: ${PROJECT_NAME}${NC}"
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# ========================================
# 1. CREATE DIRECTORY STRUCTURE
# ========================================
echo -e "${GREEN}Creating directory structure...${NC}"

mkdir -p backend/{agents,exchanges,data,risk,db/migrations,api/routes,scheduler,utils,tests}
mkdir -p frontend/{app/{leaderboard,models/[id]},components/Layout,lib,types,public}
mkdir -p docker scripts docs logs config/{grafana/{dashboards,datasources}}

# Create __init__.py files
touch backend/__init__.py
touch backend/agents/__init__.py
touch backend/exchanges/__init__.py
touch backend/data/__init__.py
touch backend/risk/__init__.py
touch backend/db/__init__.py
touch backend/api/__init__.py
touch backend/api/routes/__init__.py
touch backend/scheduler/__init__.py
touch backend/utils/__init__.py
touch backend/tests/__init__.py

# ========================================
# 2. CREATE .GITIGNORE
# ========================================
echo -e "${GREEN}Creating .gitignore...${NC}"

cat > .gitignore << 'GITIGNORE_EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv
pip-log.txt
.pytest_cache/
.coverage
htmlcov/

# Environment
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Database
*.db
*.sqlite3

# Node
node_modules/
.next/
out/
build/
dist/

# Docker
*.pid

# Misc
.cache/
tmp/
GITIGNORE_EOF

# ========================================
# 3. CREATE .ENV.EXAMPLE
# ========================================
echo -e "${GREEN}Creating .env.example...${NC}"

cat > .env.example << 'ENV_EOF'
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/alpha_arena
DATABASE_POOL_SIZE=20

# Redis
REDIS_URL=redis://localhost:6379/0

# API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROK_API_KEY=your_grok_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
QWEN_API_KEY=your_qwen_api_key_here

# Hyperliquid
HYPERLIQUID_API_KEY=your_hyperliquid_api_key_here
HYPERLIQUID_SECRET=your_hyperliquid_secret_here
HYPERLIQUID_TESTNET=true

# Trading Configuration
STARTING_CAPITAL=10000
MAX_POSITION_SIZE=5000
MAX_LEVERAGE=10
DAILY_LOSS_LIMIT=1000
CYCLE_INTERVAL_HOURS=4

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090

# Competition
COMPETITION_START_DATE=2025-01-01T00:00:00Z
COMPETITION_END_DATE=2025-11-03T17:00:00Z
ENV_EOF

# Copy to .env for convenience
cp .env.example .env

# ========================================
# 4. CREATE REQUIREMENTS.TXT
# ========================================
echo -e "${GREEN}Creating backend/requirements.txt...${NC}"

cat > backend/requirements.txt << 'REQ_EOF'
# AI/LLM SDKs
anthropic==0.40.0
openai==1.54.0
google-generativeai==0.8.0

# Web Framework
fastapi==0.115.0
uvicorn[standard]==0.32.0
websockets==13.1
python-multipart==0.0.12

# Database
sqlalchemy==2.0.35
alembic==1.13.3
psycopg2-binary==2.9.9
asyncpg==0.29.0

# Cache & Queue
redis==5.2.0
celery==5.4.0

# HTTP Client
httpx==0.27.2
aiohttp==3.10.10

# Data Processing
pandas==2.2.3
numpy==2.1.3

# Exchange/Trading
ccxt==4.4.24

# Scheduling
apscheduler==3.10.4

# Configuration
python-dotenv==1.0.1
pydantic==2.9.2
pydantic-settings==2.6.0

# Logging & Monitoring
structlog==24.4.0
prometheus-client==0.21.0

# Testing
pytest==8.3.3
pytest-asyncio==0.24.0
pytest-mock==3.14.0
httpx-mock==0.17.3

# Utilities
python-dateutil==2.9.0
pytz==2024.2
REQ_EOF

# ========================================
# 5. CREATE DOCKER-COMPOSE.YML
# ========================================
echo -e "${GREEN}Creating docker-compose.yml...${NC}"

cat > docker-compose.yml << 'DOCKER_COMPOSE_EOF'
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg16
    container_name: alpha-arena-db
    environment:
      POSTGRES_DB: alpha_arena
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: trading_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trading_user -d alpha_arena"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - alpha-arena-network

  redis:
    image: redis:7-alpine
    container_name: alpha-arena-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    networks:
      - alpha-arena-network

  trading-engine:
    build:
      context: ./backend
      dockerfile: ../docker/backend.Dockerfile
    container_name: alpha-arena-engine
    env_file:
      - .env
    environment:
      DATABASE_URL: postgresql://trading_user:trading_password@postgres:5432/alpha_arena
      REDIS_URL: redis://redis:6379/0
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./backend:/app
      - ./logs:/app/logs
    command: python main.py
    restart: unless-stopped
    networks:
      - alpha-arena-network

  api:
    build:
      context: ./backend
      dockerfile: ../docker/backend.Dockerfile
    container_name: alpha-arena-api
    env_file:
      - .env
    environment:
      DATABASE_URL: postgresql://trading_user:trading_password@postgres:5432/alpha_arena
      REDIS_URL: redis://redis:6379/0
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    restart: unless-stopped
    networks:
      - alpha-arena-network

  frontend:
    build:
      context: ./frontend
      dockerfile: ../docker/frontend.Dockerfile
    container_name: alpha-arena-frontend
    environment:
      NEXT_PUBLIC_API_URL: http://api:8000
      NEXT_PUBLIC_WS_URL: ws://api:8000
    depends_on:
      - api
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
      - /app/.next
    command: npm run dev
    restart: unless-stopped
    networks:
      - alpha-arena-network

  prometheus:
    image: prom/prometheus:latest
    container_name: alpha-arena-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - alpha-arena-network

  grafana:
    image: grafana/grafana:latest
    container_name: alpha-arena-grafana
    ports:
      - "3001:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_INSTALL_PLUGINS: grafana-clock-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./config/grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    networks:
      - alpha-arena-network

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  alpha-arena-network:
    driver: bridge
DOCKER_COMPOSE_EOF

# ========================================
# 6. CREATE DOCKERFILES
# ========================================
echo -e "${GREEN}Creating Dockerfiles...${NC}"

cat > docker/backend.Dockerfile << 'BACKEND_DOCKERFILE_EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["python", "main.py"]
BACKEND_DOCKERFILE_EOF

cat > docker/frontend.Dockerfile << 'FRONTEND_DOCKERFILE_EOF'
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
FRONTEND_DOCKERFILE_EOF

# ========================================
# 7. CREATE PROMETHEUS CONFIG
# ========================================
echo -e "${GREEN}Creating config files...${NC}"

cat > config/prometheus.yml << 'PROMETHEUS_EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'alpha-arena-api'
    static_configs:
      - targets: ['api:8000']
  
  - job_name: 'alpha-arena-engine'
    static_configs:
      - targets: ['trading-engine:9090']
PROMETHEUS_EOF

# ========================================
# 8. CREATE DATABASE INIT SCRIPT
# ========================================
echo -e "${GREEN}Creating database scripts...${NC}"

cat > scripts/init_db.sql << 'INIT_DB_EOF'
-- Initialize TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_model_id ON trades(model_id);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);
INIT_DB_EOF

# ========================================
# 9. CREATE FRONTEND PACKAGE.JSON
# ========================================
echo -e "${GREEN}Creating frontend/package.json...${NC}"

cat > frontend/package.json << 'PACKAGE_JSON_EOF'
{
  "name": "alpha-arena-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "^14.2.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "typescript": "^5.6.0",
    "recharts": "^2.12.0",
    "lightweight-charts": "^4.2.0",
    "socket.io-client": "^4.8.0",
    "tailwindcss": "^3.4.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "eslint": "^8.0.0",
    "eslint-config-next": "^14.2.0"
  }
}
PACKAGE_JSON_EOF

# ========================================
# 10. CREATE README FILES
# ========================================
echo -e "${GREEN}Creating README files...${NC}"

cat > README.md << 'README_EOF'
# Alpha Arena - AI Trading Competition

Autonomous AI trading system where AI models compete in real crypto markets.

## Quick Start

```bash
# 1. Edit .env with your API keys
nano .env

# 2. Start with Docker
docker-compose up -d

# 3. Check logs
docker-compose logs -f trading-engine

# 4. Access dashboard
# Frontend: http://localhost:3000
# API: http://localhost:8000/docs
# Grafana: http://localhost:3001
```

## Manual Setup

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py

# Frontend
cd frontend
npm install
npm run dev
```

## Documentation

- See QUICKSTART.md for detailed setup
- See ARCHITECTURE.md for system design
- See CHECKLIST.md for implementation guide

## API Keys Required

- Anthropic (Claude)
- OpenAI (GPT)
- Google (Gemini)
- Hyperliquid (Exchange)

Add all keys to .env file.

## License

MIT
README_EOF

cat > backend/README.md << 'BACKEND_README_EOF'
# Backend - Trading Engine

Python-based autonomous trading engine.

## Structure

- agents/ - AI trading agents
- exchanges/ - Exchange integrations
- data/ - Market data & indicators
- risk/ - Risk management
- db/ - Database layer
- api/ - FastAPI application

## Running

```bash
pip install -r requirements.txt
python main.py
```
BACKEND_README_EOF

cat > frontend/README.md << 'FRONTEND_README_EOF'
# Frontend - Trading Dashboard

Next.js dashboard for real-time trading monitoring.

## Running

```bash
npm install
npm run dev
```

Open http://localhost:3000
FRONTEND_README_EOF

# ========================================
# 11. FINAL MESSAGE
# ========================================
echo ""
echo -e "${GREEN}✨ Setup Complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. cd $PROJECT_NAME"
echo "2. Edit .env with your API keys:"
echo "   nano .env"
echo ""
echo "3. Start with Docker:"
echo "   docker-compose up -d"
echo ""
echo "4. OR run manually:"
echo "   cd backend && python -m venv venv && source venv/bin/activate"
echo "   pip install -r requirements.txt"
echo "   python main.py"
echo ""
echo -e "${GREEN}Access points:${NC}"
echo "- Frontend: http://localhost:3000"
echo "- API Docs: http://localhost:8000/docs"
echo "- Grafana: http://localhost:3001 (admin/admin)"
echo ""
echo -e "${GREEN}Download the core Python files from the generated outputs!${NC}"
echo ""
echo "Happy coding! 🚀"