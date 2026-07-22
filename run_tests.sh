#!/bin/bash
# CodKing Test Runner
# Run comprehensive test suite with various configurations

set -e

echo "====================================="
echo "CodKing Test Suite"
echo "====================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse arguments
FAST_ONLY=false
COVERAGE=true
VERBOSE=false
MARKERS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            FAST_ONLY=true
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -m|--markers)
            MARKERS="-m $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_tests.sh [--fast] [--no-coverage] [--verbose] [-m MARKERS]"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest tests/"

if [ "$FAST_ONLY" = true ]; then
    echo -e "${YELLOW}Running fast tests only (excluding slow and gpu tests)${NC}"
    PYTEST_CMD="$PYTEST_CMD -m 'not slow and not gpu'"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src/codking --cov-report=html --cov-report=term-missing"
fi

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -vv"
else
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD $MARKERS"
fi

# Run tests
echo -e "${GREEN}Running command: $PYTEST_CMD${NC}"
echo ""

$PYTEST_CMD

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"

    if [ "$COVERAGE" = true ]; then
        echo ""
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
    fi

    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
