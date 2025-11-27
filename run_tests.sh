#!/bin/bash

# Test runner script for Chisel Image Processing Library
# This script runs all tests with coverage reporting

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print header
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Running Tests for Chisel Library${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated${NC}"
    echo -e "${YELLOW}Consider running: source venv/bin/activate${NC}\n"
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo -e "${YELLOW}Please install dependencies: pip install -r requirements.txt${NC}"
    exit 1
fi

# Run tests with coverage
echo -e "${YELLOW}Running pytest with coverage...${NC}\n"
pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

TEST_EXIT_CODE=$?

# Print results
echo -e "\n${YELLOW}========================================${NC}"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
else
    echo -e "${RED}✗ Tests failed with exit code $TEST_EXIT_CODE${NC}"
fi
echo -e "${YELLOW}========================================${NC}\n"

exit $TEST_EXIT_CODE
