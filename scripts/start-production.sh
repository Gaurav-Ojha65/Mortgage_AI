#!/bin/bash

# Mortgage AI Production Startup Script
# Usage: ./start-production.sh [command]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║           Mortgage AI - Production System                  ║"
    echo "║                                                            ║"
    echo "║  ML Ensemble + SHAP + Drift Detection + Monitoring         ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed${NC}"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Docker Compose is not installed${NC}"
        exit 1
    fi

    echo -e "${GREEN}Prerequisites OK${NC}"
}

# Create required directories
setup_directories() {
    echo -e "${YELLOW}Setting up directories...${NC}"
    mkdir -p monitoring/grafana/dashboards monitoring/grafana/datasources
    mkdir -p docker
    mkdir -p models
    echo -e "${GREEN}Directories created${NC}"
}

# Pull and build images
build_images() {
    echo -e "${YELLOW}Building Docker images...${NC}"
    docker-compose build --parallel
    echo -e "${GREEN}Images built successfully${NC}"
}

# Start all services
start_services() {
    echo -e "${YELLOW}Starting services...${NC}"
    docker-compose up -d

    # Wait for services to be healthy
    echo -e "${YELLOW}Waiting for services to be ready...${NC}"
    sleep 10

    check_health
}

# Check service health
check_health() {
    echo -e "${YELLOW}Checking service health...${NC}"

    services=("backend" "frontend" "redis" "db" "mlflow" "prometheus" "grafana")
    all_healthy=true

    for service in "${services[@]}"; do
        if docker-compose ps "$service" | grep -q "Up"; then
            echo -e "${GREEN}  $service: Running${NC}"
        else
            echo -e "${RED}  $service: Not running${NC}"
            all_healthy=false
        fi
    done

    if [ "$all_healthy" = true ]; then
        echo -e "${GREEN}"
        echo "╔════════════════════════════════════════════════════════════╗"
        echo "║                 ALL SERVICES RUNNING                       ║"
        echo "╠════════════════════════════════════════════════════════════╣"
        echo "║  Frontend:       http://localhost                         ║"
        echo "║  API:            http://localhost:8000                      ║"
        echo "║  API Docs:       http://localhost:8000/docs               ║"
        echo "║  Grafana:        http://localhost:3000 (admin/admin)      ║"
        echo "║  MLflow:         http://localhost:5000                    ║"
        echo "║  Prometheus:     http://localhost:9090                    ║"
        echo "╚════════════════════════════════════════════════════════════╝"
        echo -e "${NC}"
    else
        echo -e "${RED}Some services are not running. Check logs with: docker-compose logs${NC}"
    fi
}

# View logs
view_logs() {
    docker-compose logs -f "$@"
}

# Stop all services
stop_services() {
    echo -e "${YELLOW}Stopping services...${NC}"
    docker-compose down
    echo -e "${GREEN}Services stopped${NC}"
}

# Clean up
cleanup() {
    echo -e "${RED}WARNING: This will remove all data including databases!${NC}"
    read -p "Are you sure? (yes/no) " confirm
    if [ "$confirm" = "yes" ]; then
        docker-compose down -v
        docker system prune -f
        echo -e "${GREEN}Cleanup complete${NC}"
    else
        echo -e "${YELLOW}Cleanup cancelled${NC}"
    fi
}

# Run tests
run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"

    # Unit tests
    echo "Running unit tests..."
    docker-compose exec backend pytest tests/ -v || true

    # Load tests
    echo "Starting load tests..."
    echo "Open http://localhost:8089 for Locust UI"
    docker-compose exec backend locust -f tests/load_test.py --host http://localhost:8000 || true
}

# Backup database
backup_database() {
    echo -e "${YELLOW}Creating database backup...${NC}"
    timestamp=$(date +%Y%m%d_%H%M%S)
    docker-compose exec -T db pg_dump -U postgres mortgage > "backup_$timestamp.sql"
    echo -e "${GREEN}Backup saved to backup_$timestamp.sql${NC}"
}

# Main command handler
case "${1:-start}" in
    start)
        print_banner
        check_prerequisites
        setup_directories
        build_images
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        start_services
        ;;
    status|health)
        check_health
        ;;
    logs)
        view_logs "${@:2}"
        ;;
    test)
        run_tests
        ;;
    backup)
        backup_database
        ;;
    cleanup)
        cleanup
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test|backup|cleanup}"
        echo ""
        echo "Commands:"
        echo "  start    - Start all services"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  status   - Check service health"
        echo "  logs     - View logs (optionally specify service)"
        echo "  test     - Run all tests"
        echo "  backup   - Backup database"
        echo "  cleanup  - Remove all containers and volumes"
        exit 1
        ;;
esac
