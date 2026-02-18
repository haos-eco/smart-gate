#!/bin/bash

echo "Setting up Git hooks..."

git config core.hooksPath .githooks

echo "✅ Git hooks installed!"
echo ""
echo "Pre-commit hook will run tests before each commit."
