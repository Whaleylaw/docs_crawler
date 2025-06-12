#!/bin/bash

# Crawl4AI Standalone Application Deployment Script
# Supports multiple deployment modes: development, staging, production

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Default values
DEPLOYMENT_MODE="production"
COMPOSE_FILE="docker-compose.yml"
ENVIRONMENT_FILE=".env"
BUILD_CACHE=true
PULL_IMAGES=true
RUN_MIGRATIONS=true
BACKUP_DATA=true
MONITORING_ENABLED=false

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${PURPLE}[DEPLOY]${NC} $1"
}

# Function to show help
show_help() {
    cat << EOF
Crawl4AI Standalone Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -m, --mode MODE          Deployment mode (development, staging, production)
    -f, --compose-file FILE  Docker compose file to use
    -e, --env-file FILE      Environment file to use
    --no-cache              Disable build cache
    --no-pull               Don't pull latest images
    --no-migration          Skip database migrations
    --no-backup             Skip data backup
    --monitoring            Enable monitoring stack
    -h, --help              Show this help message

DEPLOYMENT MODES:
    development    Local development with hot reload
    staging        Staging environment for testing
    production     Production deployment with full stack

EXAMPLES:
    $0                                    # Production deployment
    $0 -m development                     # Development mode
    $0 -m production --monitoring         # Production with monitoring
    $0 -f docker-compose.staging.yml     # Custom compose file

EOF
}

# Function to parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--mode)
                DEPLOYMENT_MODE="$2"
                shift 2
                ;;
            -f|--compose-file)
                COMPOSE_FILE="$2"
                shift 2
                ;;
            -e|--env-file)
                ENVIRONMENT_FILE="$2"
                shift 2
                ;;
            --no-cache)
                BUILD_CACHE=false
                shift
                ;;
            --no-pull)
                PULL_IMAGES=false
                shift
                ;;
            --no-migration)
                RUN_MIGRATIONS=false
                shift
                ;;
            --no-backup)
                BACKUP_DATA=false
                shift
                ;;
            --monitoring)
                MONITORING_ENABLED=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Function to validate deployment mode
validate_deployment_mode() {
    case $DEPLOYMENT_MODE in
        development|staging|production)
            log_info "Deployment mode: $DEPLOYMENT_MODE"
            ;;
        *)
            log_error "Invalid deployment mode: $DEPLOYMENT_MODE"
            log_error "Valid modes: development, staging, production"
            exit 1
            ;;
    esac
}

# Function to check prerequisites
check_prerequisites() {
    log_header "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    # Determine Docker Compose command
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
    else
        DOCKER_COMPOSE="docker compose"
    fi
    
    log_success "Prerequisites check passed"
}

# Function to validate environment file
validate_environment() {
    log_header "Validating environment configuration..."
    
    if [[ ! -f "$ENVIRONMENT_FILE" ]]; then
        log_error "Environment file not found: $ENVIRONMENT_FILE"
        
        if [[ -f ".env.example" ]]; then
            log_info "Creating environment file from template..."
            cp .env.example "$ENVIRONMENT_FILE"
            log_warning "Please edit $ENVIRONMENT_FILE with your configuration"
            log_warning "Required variables: SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY"
            exit 1
        else
            log_error "No environment template found. Please create $ENVIRONMENT_FILE"
            exit 1
        fi
    fi
    
    # Check for required variables
    source "$ENVIRONMENT_FILE" 2>/dev/null || true
    
    local required_vars=()
    local missing_vars=()
    
    # Define required variables based on deployment mode
    case $DEPLOYMENT_MODE in
        production)
            required_vars=("SUPABASE_URL" "SUPABASE_SERVICE_KEY" "OPENAI_API_KEY")
            ;;
        staging)
            required_vars=("SUPABASE_URL" "SUPABASE_SERVICE_KEY" "OPENAI_API_KEY")
            ;;
        development)
            required_vars=("OPENAI_API_KEY")
            ;;
    esac
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            log_error "  - $var"
        done
        log_error "Please update $ENVIRONMENT_FILE with the required values"
        exit 1
    fi
    
    log_success "Environment validation passed"
}

# Function to validate compose file
validate_compose_file() {
    log_header "Validating Docker Compose configuration..."
    
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    # Validate compose file syntax
    if ! $DOCKER_COMPOSE -f "$COMPOSE_FILE" config &> /dev/null; then
        log_error "Invalid Docker Compose file: $COMPOSE_FILE"
        $DOCKER_COMPOSE -f "$COMPOSE_FILE" config
        exit 1
    fi
    
    log_success "Docker Compose validation passed"
}

# Function to backup existing data
backup_data() {
    if [[ "$BACKUP_DATA" == "true" && "$DEPLOYMENT_MODE" == "production" ]]; then
        log_header "Creating data backup..."
        
        local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"
        
        # Backup volumes if they exist
        if docker volume ls | grep -q crawl4ai; then
            log_info "Backing up application data..."
            
            # Create backup container
            docker run --rm \
                -v crawl4ai_app_data:/data:ro \
                -v "$(pwd)/$backup_dir":/backup \
                alpine:latest \
                tar czf /backup/app_data.tar.gz -C /data .
            
            # Backup database if using local PostgreSQL
            if docker ps | grep -q crawl4ai-postgres; then
                log_info "Backing up database..."
                docker exec crawl4ai-postgres pg_dump -U crawl4ai crawl4ai > "$backup_dir/database.sql"
            fi
            
            log_success "Data backup completed: $backup_dir"
        else
            log_info "No existing data found, skipping backup"
        fi
    fi
}

# Function to pull latest images
pull_images() {
    if [[ "$PULL_IMAGES" == "true" ]]; then
        log_header "Pulling latest images..."
        
        # Compose profiles for monitoring
        local profiles=""
        if [[ "$MONITORING_ENABLED" == "true" ]]; then
            profiles="--profile monitoring"
        fi
        
        $DOCKER_COMPOSE -f "$COMPOSE_FILE" $profiles pull
        log_success "Images pulled successfully"
    fi
}

# Function to build application
build_application() {
    log_header "Building application..."
    
    local build_args=""
    if [[ "$BUILD_CACHE" == "false" ]]; then
        build_args="--no-cache"
    fi
    
    # Set build arguments
    export BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    export VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    export VERSION=${VERSION:-"1.0.0"}
    
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" build $build_args
    log_success "Application built successfully"
}

# Function to start services
start_services() {
    log_header "Starting services..."
    
    # Compose profiles for monitoring
    local profiles=""
    if [[ "$MONITORING_ENABLED" == "true" ]]; then
        profiles="--profile monitoring"
    fi
    
    # Start services based on deployment mode
    case $DEPLOYMENT_MODE in
        development)
            $DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d crawl4ai-app redis
            ;;
        staging|production)
            $DOCKER_COMPOSE -f "$COMPOSE_FILE" $profiles up -d
            ;;
    esac
    
    log_success "Services started successfully"
}

# Function to wait for services to be ready
wait_for_services() {
    log_header "Waiting for services to be ready..."
    
    # Wait for main application
    local max_attempts=60
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -sf http://localhost:${STREAMLIT_PORT:-8501}/_stcore/health &> /dev/null; then
            log_success "Application is ready!"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Application failed to start within expected time"
            log_info "Checking application logs:"
            $DOCKER_COMPOSE -f "$COMPOSE_FILE" logs crawl4ai-app
            exit 1
        fi
        
        log_info "Waiting for application... (attempt $attempt/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done
}

# Function to run database migrations
run_migrations() {
    if [[ "$RUN_MIGRATIONS" == "true" ]]; then
        log_header "Running database migrations..."
        
        # Run migrations inside the application container
        $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T crawl4ai-app python -c "
import sys
sys.path.append('/app')
from components.supabase_integration import initialize_database

try:
    initialize_database()
    print('✓ Database initialization completed')
except Exception as e:
    print(f'✗ Database initialization failed: {e}')
    sys.exit(1)
"
        
        log_success "Database migrations completed"
    fi
}

# Function to run health checks
run_health_checks() {
    log_header "Running health checks..."
    
    # Check main application
    if curl -sf http://localhost:${STREAMLIT_PORT:-8501}/_stcore/health &> /dev/null; then
        log_success "✓ Streamlit application is healthy"
    else
        log_error "✗ Streamlit application health check failed"
    fi
    
    # Check API server
    if curl -sf http://localhost:${API_PORT:-8000}/health &> /dev/null; then
        log_success "✓ API server is healthy"
    else
        log_warning "⚠ API server health check failed (may not be enabled)"
    fi
    
    # Check Redis
    if docker exec crawl4ai-redis redis-cli ping &> /dev/null; then
        log_success "✓ Redis is healthy"
    else
        log_warning "⚠ Redis health check failed"
    fi
    
    # Check PostgreSQL (if running)
    if docker ps | grep -q crawl4ai-postgres; then
        if docker exec crawl4ai-postgres pg_isready -U crawl4ai &> /dev/null; then
            log_success "✓ PostgreSQL is healthy"
        else
            log_warning "⚠ PostgreSQL health check failed"
        fi
    fi
}

# Function to show deployment summary
show_deployment_summary() {
    log_header "Deployment Summary"
    
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    DEPLOYMENT COMPLETED                     ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Deployment Mode:${NC} $DEPLOYMENT_MODE"
    echo -e "${BLUE}Compose File:${NC} $COMPOSE_FILE"
    echo -e "${BLUE}Environment File:${NC} $ENVIRONMENT_FILE"
    echo ""
    echo -e "${GREEN}🌐 Application URLs:${NC}"
    echo -e "   Streamlit UI: http://localhost:${STREAMLIT_PORT:-8501}"
    
    if [[ "$DEPLOYMENT_MODE" != "development" ]]; then
        echo -e "   API Server: http://localhost:${API_PORT:-8000}"
        echo -e "   API Docs: http://localhost:${API_PORT:-8000}/docs"
    fi
    
    if [[ "$MONITORING_ENABLED" == "true" ]]; then
        echo -e "   Grafana: http://localhost:${GRAFANA_PORT:-3000}"
        echo -e "   Prometheus: http://localhost:${PROMETHEUS_PORT:-9090}"
    fi
    
    echo ""
    echo -e "${GREEN}📋 Useful Commands:${NC}"
    echo -e "   View logs: $DOCKER_COMPOSE -f $COMPOSE_FILE logs"
    echo -e "   Stop services: $DOCKER_COMPOSE -f $COMPOSE_FILE down"
    echo -e "   Restart services: $DOCKER_COMPOSE -f $COMPOSE_FILE restart"
    echo -e "   Update application: $0 -m $DEPLOYMENT_MODE"
    echo ""
}

# Function to handle cleanup on exit
cleanup() {
    if [[ $? -ne 0 ]]; then
        log_error "Deployment failed. Cleaning up..."
        $DOCKER_COMPOSE -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    fi
}

# Main deployment function
main() {
    # Set up cleanup trap
    trap cleanup EXIT
    
    log_header "Starting Crawl4AI Standalone Deployment"
    log_info "Script started at $(date)"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Validation steps
    validate_deployment_mode
    check_prerequisites
    validate_environment
    validate_compose_file
    
    # Deployment steps
    backup_data
    pull_images
    build_application
    start_services
    wait_for_services
    run_migrations
    run_health_checks
    
    # Show summary
    show_deployment_summary
    
    log_success "Deployment completed successfully!"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi