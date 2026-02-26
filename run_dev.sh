#!/usr/bin/bash
# Development startup script for Open-CITE GUI

echo "🚀 Starting Open-CITE GUI in Development Mode"
echo ""

# Activate virtual environment
source venv/bin/activate

# Enable persistence
export OPENCITE_PERSISTENCE_ENABLED=true
export OPENCITE_DATABASE_URL="sqlite:///./opencite.db"

# Export development environment variables (optional)
export FLASK_ENV=development
export FLASK_DEBUG=1
export OPENCITE_LOG_LEVEL=DEBUG

# Launch GUI with debug mode
opencite gui --debug --host 127.0.0.1 --port 5000
