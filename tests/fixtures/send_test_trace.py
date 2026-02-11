#!/usr/bin/env python3
"""
Send a test OTLP trace to OpenCITE for testing the UI.
This simulates an AI tool making API calls that will appear in the UI.
"""

import requests
import json
import sys

# Default OTLP endpoint - change if your app runs on a different port
OTLP_ENDPOINT = "http://localhost:4318/v1/traces"

def send_test_trace(tool_name: str = "test-tool", model_name: str = "openai/gpt-4", num_calls: int = 1):
    """
    Send a test OTLP trace to the OpenCITE receiver.
    
    Args:
        tool_name: Name of the tool making the API call
        model_name: Model being used (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
        num_calls: Number of API calls to simulate
    """
    print(f"Sending {num_calls} test trace(s) to {OTLP_ENDPOINT}...")
    print(f"Tool: {tool_name}")
    print(f"Model: {model_name}\n")
    
    for i in range(num_calls):
        # Create a trace that mimics OpenRouter/OpenAI API calls
        trace_data = {
            "resourceSpans": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": tool_name}},
                        {"key": "service.version", "value": {"stringValue": "1.0.0"}}
                    ]
                },
                "scopeSpans": [{
                    "scope": {
                        "name": "openai",
                        "version": "1.0.0"
                    },
                    "spans": [{
                        "traceId": f"abc123def456{i:04d}",
                        "spanId": f"123456{i:04d}",
                        "name": "chat.completions",
                        "kind": "SPAN_KIND_CLIENT",
                        "startTimeUnixNano": str(1700000000000000000 + i * 1000000000),
                        "endTimeUnixNano": str(1700000000000000000 + i * 1000000000 + 500000000),
                        "attributes": [
                            {"key": "gen_ai.request.model", "value": {"stringValue": model_name}},
                            {"key": "gen_ai.system", "value": {"stringValue": "openrouter"}},
                            {"key": "gen_ai.request.type", "value": {"stringValue": "chat"}},
                            {"key": "http.method", "value": {"stringValue": "POST"}},
                            {"key": "http.url", "value": {"stringValue": "https://openrouter.ai/api/v1/chat/completions"}},
                            {"key": "http.status_code", "value": {"intValue": "200"}}
                        ],
                        "status": {
                            "code": "STATUS_CODE_OK"
                        }
                    }]
                }]
            }]
        }
        
        try:
            response = requests.post(
                OTLP_ENDPOINT,
                json=trace_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"✅ Trace {i+1}/{num_calls} sent successfully")
            else:
                print(f"⚠️  Trace {i+1}/{num_calls} returned status {response.status_code}: {response.text[:100]}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection Error: Could not connect to {OTLP_ENDPOINT}")
            print(f"   Make sure:")
            print(f"   1. The Flask app is running")
            print(f"   2. OpenTelemetry plugin is configured in the UI")
            print(f"   3. The OTLP receiver is started (check status in UI)")
            return False
        except Exception as e:
            print(f"❌ Error sending trace {i+1}: {e}")
            return False
    
    print(f"\n✅ All {num_calls} trace(s) sent!")
    print(f"\nNext steps:")
    print(f"1. Go to the OpenCITE UI: http://127.0.0.1:5000")
    print(f"2. Click 'Discover Assets' or wait a few seconds for auto-refresh")
    print(f"3. You should see '{tool_name}' in the Tools tab")
    print(f"4. You should see '{model_name}' in the Models tab")
    return True


def main():
    """Main function with command-line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Send test OTLP traces to OpenCITE for UI testing"
    )
    parser.add_argument(
        "--tool",
        default="test-tool",
        help="Name of the tool (default: test-tool)"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4",
        help="Model name (default: openai/gpt-4). Examples: openai/gpt-4, anthropic/claude-3-opus, google/gemini-pro"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of traces to send (default: 1)"
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:4318/v1/traces",
        help="OTLP endpoint URL (default: http://localhost:4318/v1/traces)"
    )
    
    args = parser.parse_args()
    
    global OTLP_ENDPOINT
    OTLP_ENDPOINT = args.endpoint
    
    success = send_test_trace(
        tool_name=args.tool,
        model_name=args.model,
        num_calls=args.count
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

