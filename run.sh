#!/bin/bash

# Real-time Voice Assistant - Quick Start Script

set -e

echo "=========================================="
echo "Real-time Voice Assistant"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "Please restart your terminal and run this script again."
    exit 1
fi

# Check if dependencies are installed
if [ ! -f ".venv/bin/python" ]; then
    echo "Setting up Python environment with uv..."
    uv sync
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration"
    echo ""
fi

# Check if Ollama is running
echo "Checking Ollama connection..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Warning: Cannot connect to Ollama at http://localhost:11434"
    echo "Please make sure Ollama is running:"
    echo "  1. Install Ollama: https://ollama.ai"
    echo "  2. Start Ollama: ollama serve"
    echo "  3. Pull model: ollama pull midm-2.0-q8_0:base"
    echo ""
    echo "Press Enter to continue anyway or Ctrl+C to abort..."
    read
else
    echo "✓ Ollama is running"

    # Check if model is available
    if ! curl -s http://localhost:11434/api/tags | grep -q "midm-2.0-q8_0:base"; then
        echo "⚠️  Model 'midm-2.0-q8_0:base' not found"
        echo "Pulling model... (this may take a few minutes)"
        ollama pull midm-2.0-q8_0:base
    else
        echo "✓ Model midm-2.0-q8_0:base is available"
    fi
fi

# Start server
echo ""
echo "Starting server..."
echo "Open http://localhost:8000/client in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python -m server.main
