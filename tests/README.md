# OpenCITE Test Suite

This directory contains the test suite for OpenCITE.

## Structure

```
tests/
├── __init__.py              # Package initialization
├── test_api.py              # API endpoint tests
├── UI_TESTING_GUIDE.md      # Complete guide for testing the UI
├── fixtures/                # Test fixtures and utilities
│   ├── __init__.py
│   └── send_test_trace.py   # Utility to send test OTLP traces
└── README.md                # This file
```

## Running Tests

### API Tests

Test all API endpoints:

```bash
python tests/test_api.py
```

Test with custom port:

```bash
python tests/test_api.py 8080
```

### Sending Test Traces

Send test OTLP traces to test the UI:

```bash
# Basic usage
python tests/fixtures/send_test_trace.py

# With options
python tests/fixtures/send_test_trace.py --tool "my-app" --model "openai/gpt-4" --count 5
```

See `tests/fixtures/send_test_trace.py --help` for all options.

## Test Organization

- **`test_api.py`**: Integration tests for the Flask API endpoints
- **`UI_TESTING_GUIDE.md`**: Complete guide for testing the web UI, sending traces, and viewing results
- **`fixtures/send_test_trace.py`**: Utility script to generate test OTLP traces for UI testing

## Documentation

For detailed UI testing instructions, see **[UI_TESTING_GUIDE.md](UI_TESTING_GUIDE.md)**.

## Adding New Tests

When adding new tests:

1. Create test files following the naming convention: `test_*.py`
2. Place test utilities in `fixtures/` directory
3. Use descriptive test function names
4. Include docstrings explaining what each test validates

## Requirements

Tests require:
- `requests` library (for HTTP testing)
- Flask app running on localhost:5000 (or specified port)

