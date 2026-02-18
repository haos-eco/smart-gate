#!/bin/bash

cd /app

# Run tests
pytest tests/ \
    --verbose \
    --cov=. \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-exclude=tests/

# Open coverage report
echo "Coverage report generated in htmlcov/index.html"
