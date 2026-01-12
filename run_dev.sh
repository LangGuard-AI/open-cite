#!/usr/bin/bash
# Development startup script for OpenCITE GUI

echo "🚀 Starting OpenCITE GUI in Development Mode"
echo ""

# Activate virtual environment
source venv/bin/activate

# Export development environment variables (optional)
export FLASK_ENV=development
export FLASK_DEBUG=1

# Launch GUI with debug mode
opencite gui --debug --host 127.0.0.1 --port 5000
