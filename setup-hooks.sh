#!/bin/bash

echo "Setting up Git hooks..."

chmod +x .githooks/*

git config core.hooksPath .githooks

echo "✅ Git hooks installed!"
echo ""
echo "Pre-commit hook will run tests before each commit."
