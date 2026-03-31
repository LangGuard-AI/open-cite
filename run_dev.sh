#!/usr/bin/bash
# Development startup script for Open-CITE GUI

echo "🚀 Starting Open-CITE GUI in Development Mode"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    read -p "Virtual environment not found. Create it and install packages? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        python3 -m venv venv
        source venv/bin/activate
        pip install -e .
    else
        echo "Cannot continue without a virtual environment."
        exit 1
    fi
else
    source venv/bin/activate

    # Check if the package is installed
    if ! pip show open-cite &>/dev/null; then
        read -p "Packages not installed. Install them now? [y/N] " answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            pip install -e .
        else
            echo "Cannot continue without packages installed."
            exit 1
        fi
    fi
fi

# Enable persistence
export OPENCITE_PERSISTENCE_ENABLED=true
export OPENCITE_DATABASE_URL="sqlite:///./opencite.db"

# Export development environment variables (optional)
export FLASK_ENV=development
export FLASK_DEBUG=1
export OPENCITE_LOG_LEVEL=DEBUG

# Launch GUI with debug mode
opencite gui --debug --host 127.0.0.1 --port 5000
