#!/bin/bash

cd smart_gate/app

pytest tests/ \
    --verbose \
    --cov=. \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-exclude=tests/

echo "Coverage report generated in htmlcov/index.html"
