# 🕷️ Crawl4AI Standalone Application

A comprehensive web crawling and RAG (Retrieval Augmented Generation) solution built with Streamlit, featuring advanced AI-powered content processing, vector search, and real-time monitoring.

![Crawl4AI Standalone](https://img.shields.io/badge/Version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 🌟 Features

### 🕷️ **Advanced Web Crawling**
- **Smart URL Detection**: Automatic sitemap discovery and intelligent link following
- **Concurrent Processing**: Multi-threaded crawling with configurable concurrency
- **Content Extraction**: Intelligent markdown conversion with metadata preservation
- **Recursive Crawling**: Configurable depth limits and domain filtering
- **Respectful Crawling**: Built-in rate limiting and robots.txt compliance

### 🧠 **AI-Powered RAG Strategies**
- **Vector Embeddings**: OpenAI text-embedding-3-small/large support
- **Contextual Embeddings**: AI-enhanced chunk context for better retrieval
- **Agentic RAG**: Automated code example extraction with AI summaries
- **Cross-encoder Reranking**: Advanced result relevance optimization
- **Hybrid Search**: Combined vector and keyword search capabilities

### 🗄️ **Enterprise Database Integration**
- **Supabase Integration**: Full-featured PostgreSQL with vector extensions
- **Vector Storage**: Optimized pgvector integration for similarity search
- **Project Management**: Multi-tenant project organization
- **Data Analytics**: Comprehensive content analysis and metrics

### 🔍 **Advanced Search Interface**
- **Semantic Search**: Vector similarity with configurable thresholds
- **Advanced Filtering**: Domain, date, content type, and metadata filters
- **AI Reranking**: Intelligent result optimization
- **Search History**: Persistent search tracking and analytics
- **Export Capabilities**: CSV, JSON, and formatted result exports

### 📊 **Content Analysis & Insights**
- **Quality Scoring**: AI-powered content quality assessment
- **Duplicate Detection**: Intelligent content deduplication
- **Analytics Dashboard**: Comprehensive corpus analysis and visualization
- **Document Explorer**: Interactive content browsing and preview
- **Export Tools**: Flexible data export in multiple formats

### 📈 **Real-time Monitoring**
- **System Metrics**: CPU, memory, disk, and network monitoring
- **Application Metrics**: Crawl jobs, search performance, and API usage
- **Alert System**: Configurable alerts with webhook notifications
- **Performance Analytics**: Historical analysis and trend detection
- **Health Monitoring**: Comprehensive system health dashboards

### 🔌 **REST API & Webhooks**
- **Full REST API**: Complete programmatic access to all features
- **Webhook Support**: Real-time event notifications with retry logic
- **API Key Management**: Secure authentication with rate limiting
- **Interactive Documentation**: Swagger/OpenAPI documentation
- **SDK Examples**: Python, JavaScript, and cURL examples

### ⚙️ **Advanced Configuration**
- **Environment Management**: Comprehensive configuration system
- **Model Selection**: Multiple AI model provider support
- **Performance Tuning**: Configurable processing parameters
- **Security Settings**: Comprehensive security configuration
- **Import/Export**: Configuration backup and restoration

### 🐳 **Production-Ready Deployment**
- **Docker Containerization**: Multi-stage optimized builds
- **Docker Compose**: Complete stack orchestration
- **Health Checks**: Comprehensive service monitoring
- **Graceful Shutdown**: Proper cleanup and signal handling
- **Production Deployment**: Nginx, Redis, monitoring stack included

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose**: Required for containerized deployment
- **Python 3.11+**: For local development
- **Supabase Account**: For database and vector storage
- **OpenAI API Key**: For AI-powered features

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/crawl4ai-standalone.git
cd crawl4ai-standalone
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (required)
nano .env
```

**Required Environment Variables:**
```bash
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
OPENAI_API_KEY=your_openai_api_key
```

### 3. Deploy with Docker

```bash
# Make deploy script executable
chmod +x deploy.sh

# Deploy in production mode
./deploy.sh

# Or deploy in development mode
./deploy.sh -m development
```

### 4. Access the Application

- **Streamlit UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Endpoint**: http://localhost:8000

## 📖 Detailed Setup Guide

### Local Development Setup

1. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Run Application**
```bash
streamlit run app.py
```

### Production Deployment

1. **Prepare Server**
```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER
```

2. **Configure Environment**
```bash
# Set production environment variables
export ENVIRONMENT=production
export SUPABASE_URL=your_production_url
export SUPABASE_SERVICE_KEY=your_production_key
export OPENAI_API_KEY=your_production_key
```

3. **Deploy with Monitoring**
```bash
./deploy.sh -m production --monitoring
```

4. **Access Services**
- **Application**: http://your-domain.com
- **API**: http://your-domain.com/api
- **Grafana**: http://your-domain.com:3000
- **Prometheus**: http://your-domain.com:9090

## 🎯 Usage Guide

### 1. Creating Your First Project

1. Navigate to **Project Management**
2. Click **"Create New Project"**
3. Enter project details:
   - **Name**: Your project identifier
   - **Description**: Project purpose
   - **Initial URLs**: Starting crawl URLs

### 2. Starting a Crawl Job

1. Go to **Crawl Content** page
2. Select your project
3. Configure crawl parameters:
   - **URLs**: Target websites
   - **Max Depth**: Link following depth
   - **RAG Strategies**: AI processing options
4. Click **"Start Crawling"**

### 3. Searching Content

1. Visit **Search Interface**
2. Select project and enter query
3. Configure filters:
   - **Similarity Threshold**: Result relevance
   - **Domain Filter**: Specific websites
   - **Date Range**: Content age
4. Review and export results

### 4. Analyzing Content

1. Open **Content Analysis**
2. Explore analysis tabs:
   - **Overview**: Corpus statistics
   - **Document Explorer**: Content browser
   - **Quality Analysis**: Content scoring
   - **Duplicate Detection**: Similarity analysis
   - **Analytics Dashboard**: Visual insights

### 5. Using the API

```python
import requests

# API Configuration
API_KEY = "your_api_key"
BASE_URL = "http://localhost:8000"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Create Project
project = requests.post(
    f"{BASE_URL}/projects",
    json={"name": "My Project", "description": "API Test"},
    headers=headers
).json()

# Start Crawl
crawl = requests.post(
    f"{BASE_URL}/crawl",
    json={
        "project_id": project["id"],
        "urls": ["https://example.com"],
        "rag_strategies": ["vector_embeddings"]
    },
    headers=headers
).json()

# Search Documents
results = requests.post(
    f"{BASE_URL}/search",
    json={
        "project_id": project["id"],
        "query": "machine learning",
        "limit": 10
    },
    headers=headers
).json()
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI       │    │   Crawl4AI      │
│   (Frontend)    │◄──►│   (API Server)  │◄──►│   (Engine)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Supabase      │    │   Redis         │    │   OpenAI        │
│   (Database)    │    │   (Cache)       │    │   (AI Models)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Crawl Initiation**: User triggers crawl through UI or API
2. **Content Extraction**: Crawl4AI processes web content
3. **AI Processing**: OpenAI generates embeddings and analysis
4. **Storage**: Content and vectors stored in Supabase
5. **Search & Retrieval**: Vector similarity search with AI reranking
6. **Monitoring**: Real-time metrics collection and alerting

### Technology Stack

- **Frontend**: Streamlit with custom components
- **Backend**: FastAPI with async support
- **Crawling**: Crawl4AI with Playwright/Selenium
- **Database**: Supabase (PostgreSQL + pgvector)
- **AI**: OpenAI GPT and embedding models
- **Caching**: Redis for performance optimization
- **Monitoring**: Custom metrics with Prometheus/Grafana
- **Deployment**: Docker with multi-service orchestration

## 🔧 Configuration Reference

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `SUPABASE_URL` | Supabase project URL | - | Yes |
| `SUPABASE_SERVICE_KEY` | Supabase service key | - | Yes |
| `OPENAI_API_KEY` | OpenAI API key | - | Yes |
| `MAX_CONCURRENT` | Concurrent crawl jobs | 10 | No |
| `CHUNK_SIZE` | Text chunk size | 4000 | No |
| `SIMILARITY_THRESHOLD` | Search threshold | 0.7 | No |

### RAG Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `vector_embeddings` | Basic vector search | General content retrieval |
| `contextual_embeddings` | AI-enhanced chunks | Improved context understanding |
| `agentic_rag` | Code extraction | Technical documentation |
| `cross_encoder_reranking` | Result optimization | High-precision search |

### Performance Tuning

```yaml
# High-performance configuration
MAX_CONCURRENT: 20
CHUNK_SIZE: 6000
CHUNK_OVERLAP: 300
ENABLE_CACHING: true
CACHE_TTL_MINUTES: 120
RAG_PARALLEL_WORKERS: 15
```

## 📊 Monitoring & Observability

### Built-in Metrics

- **System**: CPU, memory, disk, network usage
- **Application**: Crawl jobs, search performance, API usage
- **Business**: Content quality, duplicate rates, user activity

### Health Checks

- **Application**: Streamlit and API server status
- **Database**: Supabase connection and query performance
- **External**: OpenAI API availability and response times

### Alerting

- **Threshold Alerts**: CPU, memory, disk space warnings
- **Performance Alerts**: Slow queries, API timeouts
- **Business Alerts**: Failed crawls, high error rates

## 🧪 Testing

### Run Test Suite

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=components --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/api/
```

### Test Categories

- **Unit Tests**: Component-level testing
- **Integration Tests**: Service interaction testing
- **API Tests**: REST endpoint validation
- **E2E Tests**: Complete workflow testing

## 🔒 Security

### Authentication & Authorization

- **API Keys**: Secure token-based authentication
- **Rate Limiting**: Configurable request throttling
- **CORS**: Cross-origin request security
- **Input Validation**: Comprehensive request validation

### Data Protection

- **Encryption**: Credential encryption at rest
- **Access Control**: Project-based data isolation
- **Audit Logging**: Comprehensive activity tracking
- **Backup**: Automated data backup and retention

## 🚢 Deployment Options

### Docker Deployment

```bash
# Simple deployment
docker-compose up -d

# Production with monitoring
docker-compose --profile monitoring up -d

# Custom configuration
docker-compose -f docker-compose.prod.yml up -d
```

### Cloud Deployment

**AWS EC2**:
```bash
# Launch EC2 instance
aws ec2 run-instances --image-id ami-12345 --instance-type t3.medium

# Deploy application
scp -i key.pem deploy.sh ec2-user@instance-ip:~
ssh -i key.pem ec2-user@instance-ip './deploy.sh -m production'
```

**Google Cloud Run**:
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT/crawl4ai
gcloud run deploy --image gcr.io/PROJECT/crawl4ai --platform managed
```

### Kubernetes Deployment

```yaml
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. **Fork and Clone**
```bash
git clone https://github.com/your-username/crawl4ai-standalone.git
cd crawl4ai-standalone
```

2. **Create Feature Branch**
```bash
git checkout -b feature/amazing-feature
```

3. **Make Changes and Test**
```bash
# Run tests
pytest

# Run linting
black . && flake8
```

4. **Submit Pull Request**

### Code Style

- **Black**: Code formatting
- **Flake8**: Linting and style checks
- **Type Hints**: Required for new code
- **Docstrings**: Google-style documentation

## 🆘 Troubleshooting

### Common Issues

**1. Application Won't Start**
```bash
# Check logs
docker-compose logs crawl4ai-app

# Verify environment
cat .env | grep -E "(SUPABASE|OPENAI)"

# Test database connection
docker-compose exec crawl4ai-app python -c "from components.supabase_integration import test_connection; test_connection()"
```

**2. Crawling Fails**
```bash
# Check crawl logs
docker-compose logs | grep -i crawl

# Verify network connectivity
docker-compose exec crawl4ai-app curl -I https://target-website.com

# Check rate limiting
grep "rate limit" /app/logs/app.log
```

**3. Search Returns No Results**
```bash
# Verify embeddings
SELECT COUNT(*) FROM embeddings WHERE project_id = 'your-project-id';

# Check similarity threshold
# Try lowering similarity_threshold in search parameters

# Verify OpenAI connection
docker-compose exec crawl4ai-app python -c "import openai; print(openai.models.list())"
```

### Performance Issues

**High Memory Usage**:
- Reduce `CHUNK_SIZE` and `MAX_CONCURRENT`
- Enable `ENABLE_COMPRESSION=true`
- Monitor with `docker stats`

**Slow Searches**:
- Optimize similarity threshold
- Enable `USE_RERANKING=true`
- Add database indexes

**Crawling Timeouts**:
- Increase `REQUEST_TIMEOUT`
- Reduce `MAX_DEPTH`
- Add domain filters

## 📚 Additional Resources

- **API Documentation**: [OpenAPI Spec](http://localhost:8000/docs)
- **Crawl4AI Docs**: [Official Documentation](https://github.com/unclecode/crawl4ai)
- **Supabase Docs**: [Database Guide](https://supabase.io/docs)
- **Streamlit Docs**: [UI Framework](https://docs.streamlit.io)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Crawl4AI**: Core crawling engine
- **Streamlit**: Frontend framework
- **Supabase**: Database and backend services
- **OpenAI**: AI model integration
- **FastAPI**: API framework

---

Built with ❤️ for the developer community

**Need help?** [Open an issue](https://github.com/your-org/crawl4ai-standalone/issues) or [join our Discord](https://discord.gg/crawl4ai)