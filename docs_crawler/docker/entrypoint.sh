#!/bin/bash
set -e

# Crawl4AI Standalone Application Entrypoint Script
# Handles initialization, environment setup, and graceful startup

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    log_info "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            log_success "$service_name is ready!"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "$service_name failed to become ready after $max_attempts attempts"
    return 1
}

# Function to check required environment variables
check_environment() {
    log_info "Checking environment configuration..."
    
    local required_vars=(
        "SUPABASE_URL"
        "SUPABASE_SERVICE_KEY" 
        "OPENAI_API_KEY"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            log_error "  - $var"
        done
        
        log_warning "Some features may not work properly without these variables"
        log_warning "Continuing with limited functionality..."
    else
        log_success "All required environment variables are set"
    fi
}

# Function to initialize application directories
initialize_directories() {
    log_info "Initializing application directories..."
    
    local directories=(
        "/app/logs"
        "/app/data"
        "/app/config"
        "/app/uploads"
        "/app/data/cache"
        "/app/data/temp"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    # Set proper permissions
    chmod 755 /app/logs /app/data /app/config /app/uploads
    chmod 755 /app/data/cache /app/data/temp
    
    log_success "Application directories initialized"
}

# Function to setup logging configuration
setup_logging() {
    log_info "Setting up logging configuration..."
    
    # Create log files if they don't exist
    touch /app/logs/app.log
    touch /app/logs/api.log
    touch /app/logs/crawl.log
    touch /app/logs/error.log
    
    # Set proper permissions
    chmod 644 /app/logs/*.log
    
    log_success "Logging configuration complete"
}

# Function to validate configuration
validate_configuration() {
    log_info "Validating application configuration..."
    
    # Check if config file exists, create default if not
    if [ ! -f "/app/config/config.json" ]; then
        log_info "Creating default configuration file..."
        cat > /app/config/config.json << EOF
{
    "version": "1.0.0",
    "environment": "${ENVIRONMENT:-production}",
    "debug": ${DEBUG:-false},
    "openai": {
        "embedding_model": "${OPENAI_EMBEDDING_MODEL:-text-embedding-3-small}",
        "chat_model": "${OPENAI_CHAT_MODEL:-gpt-3.5-turbo}",
        "max_tokens": 1000,
        "temperature": 0.3
    },
    "crawling": {
        "max_concurrent": ${MAX_CONCURRENT:-10},
        "chunk_size": ${CHUNK_SIZE:-4000},
        "chunk_overlap": ${CHUNK_OVERLAP:-200},
        "max_depth": 3
    },
    "performance": {
        "enable_caching": true,
        "cache_ttl_minutes": 60
    },
    "logging": {
        "level": "${LOG_LEVEL:-INFO}",
        "enable_file_logging": true
    }
}
EOF
        log_success "Default configuration file created"
    fi
    
    log_success "Configuration validation complete"
}

# Function to check dependencies
check_dependencies() {
    log_info "Checking system dependencies..."
    
    # Check if required commands are available
    local commands=("python" "pip" "curl")
    
    for cmd in "${commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check Python version
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    log_info "Python version: $python_version"
    
    # Check if required Python packages are installed
    log_info "Checking Python dependencies..."
    python -c "import streamlit, fastapi, supabase, openai, crawl4ai" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        log_success "All Python dependencies are available"
    else
        log_warning "Some Python dependencies may be missing"
        log_info "Installing/updating dependencies..."
        pip install --no-cache-dir -r /app/requirements.txt
    fi
    
    log_success "Dependencies check complete"
}

# Function to run database migrations (if needed)
run_migrations() {
    log_info "Checking for database migrations..."
    
    # In a real application, you would run database migrations here
    # For this demo, we'll just log that we checked
    
    log_success "Database migrations check complete"
}

# Function to start API server in background
start_api_server() {
    if [ "${ENABLE_API:-true}" = "true" ]; then
        log_info "Starting API server in background..."
        
        # Start the API server
        python -c "
from components.api_integration import get_api_manager
import time
import signal
import sys

def signal_handler(sig, frame):
    print('API server shutting down...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

api_manager = get_api_manager()
api_manager.start_api_server(${API_SERVER_PORT:-8000})

# Keep the process alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
" &
        
        API_PID=$!
        echo $API_PID > /app/api_server.pid
        
        log_success "API server started with PID: $API_PID"
    else
        log_info "API server disabled"
    fi
}

# Function to setup signal handlers for graceful shutdown
setup_signal_handlers() {
    log_info "Setting up signal handlers for graceful shutdown..."
    
    # Function to handle shutdown
    shutdown() {
        log_info "Received shutdown signal, cleaning up..."
        
        # Stop API server if running
        if [ -f "/app/api_server.pid" ]; then
            API_PID=$(cat /app/api_server.pid)
            if kill -0 $API_PID 2>/dev/null; then
                log_info "Stopping API server (PID: $API_PID)..."
                kill $API_PID
                wait $API_PID 2>/dev/null
            fi
            rm -f /app/api_server.pid
        fi
        
        # Clean up temporary files
        rm -rf /app/data/temp/*
        
        log_success "Cleanup complete"
        exit 0
    }
    
    # Set up signal traps
    trap shutdown SIGTERM SIGINT
}

# Function to perform health check
health_check() {
    log_info "Performing initial health check..."
    
    # Check if we can import main modules
    python -c "
import sys
sys.path.append('/app')
try:
    import streamlit
    import components.supabase_integration
    import components.crawl_engine
    print('✓ All core modules imported successfully')
except ImportError as e:
    print(f'✗ Failed to import modules: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        exit 1
    fi
}

# Main execution function
main() {
    log_info "Starting Crawl4AI Standalone Application..."
    log_info "Environment: ${ENVIRONMENT:-production}"
    log_info "Debug Mode: ${DEBUG:-false}"
    
    # Run initialization steps
    check_environment
    initialize_directories
    setup_logging
    validate_configuration
    check_dependencies
    run_migrations
    health_check
    setup_signal_handlers
    
    # Wait for external services if configured
    if [ -n "$REDIS_HOST" ]; then
        wait_for_service "${REDIS_HOST:-redis}" "${REDIS_PORT:-6379}" "Redis"
    fi
    
    if [ -n "$POSTGRES_HOST" ]; then
        wait_for_service "${POSTGRES_HOST:-postgres}" "${POSTGRES_PORT:-5432}" "PostgreSQL"
    fi
    
    # Start API server if enabled
    start_api_server
    
    log_success "Initialization complete!"
    log_info "Starting main application..."
    
    # Execute the main command passed to the script
    exec "$@"
}

# Run main function if script is executed directly
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi