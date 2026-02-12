#!/usr/bin/env python3
"""
Test script for OpenCITE Flask application API endpoints.
Run this to verify your localhost instance is working correctly.
"""

import requests
import json
import sys

# Default base URL - change if your app runs on a different port
BASE_URL = "http://127.0.0.1:5000"

def check_endpoint(method: str, endpoint: str, data: dict = None, description: str = None, expected_status: int = 200) -> bool:
    """Test an API endpoint and print results.

    This is a helper for the manual integration test script, not a pytest test.
    """
    url = f"{BASE_URL}{endpoint}"
    print(f"\n{'='*60}")
    if description:
        print(f"Testing: {description}")
    print(f"{method.upper()} {endpoint}")
    print(f"{'='*60}")
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=5)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        print(f"Status Code: {response.status_code} (expected: {expected_status})")
        
        if response.status_code == expected_status:
            try:
                result = response.json()
                if expected_status == 200:
                    print(f"✅ Success!")
                else:
                    print(f"✅ Expected status code received!")
                print(f"Response (formatted):")
                print(json.dumps(result, indent=2))
                return True
            except json.JSONDecodeError:
                if expected_status == 200:
                    print(f"✅ Success! (Non-JSON response)")
                else:
                    print(f"✅ Expected status code received! (Non-JSON response)")
                print(f"Response: {response.text[:200]}...")
                return True
        else:
            print(f"❌ Failed with status {response.status_code} (expected {expected_status})")
            print(f"Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Could not connect to {url}")
        print(f"   Make sure the Flask app is running on {BASE_URL}")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Timeout: Request took too long")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("OpenCITE Flask Application Test Suite")
    print("="*60)
    print(f"Testing against: {BASE_URL}")
    print("\nMake sure your Flask app is running!\n")
    
    results = []
    
    # Test 1: Root page (HTML)
    print("="*60)
    print("TEST 1: Web Interface")
    print("="*60)
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print(f"✅ Web interface is accessible!")
            print(f"   Open {BASE_URL} in your browser to see the GUI")
            results.append(True)
        else:
            print(f"❌ Web interface returned status {response.status_code}")
            results.append(False)
    except Exception as e:
        print(f"❌ Could not access web interface: {e}")
        results.append(False)
    
    # Test 2: Status endpoint
    results.append(check_endpoint("GET", "/api/status", description="Get discovery status"))
    
    # Test 3: List available plugins
    results.append(check_endpoint("GET", "/api/plugins", description="List available plugins"))
    
    # Test 4: Get assets (should return 400 when no plugins configured)
    results.append(check_endpoint("GET", "/api/assets", description="Get discovered assets (expects 400 when no plugins)", expected_status=400))
    
    # Test 5: Configure OpenTelemetry plugin (no config needed)
    print("\n" + "="*60)
    print("TEST 5: Configure OpenTelemetry Plugin")
    print("="*60)
    print("This will start the OTLP trace receiver...")
    otel_config = {
        "plugins": [
            {
                "name": "opentelemetry",
                "config": {}
            }
        ]
    }
    results.append(check_endpoint("POST", "/api/plugins/configure", data=otel_config, 
                                 description="Configure OpenTelemetry plugin"))
    
    # Test 6: Check status after configuration
    results.append(check_endpoint("GET", "/api/status", description="Check status after plugin configuration"))
    
    # Test 7: Get assets again (should show OpenTelemetry endpoint info)
    results.append(check_endpoint("GET", "/api/assets", description="Get assets after plugin configuration"))
    
    # Test 8: Export data
    export_config = {
        "plugins": ["opentelemetry"]
    }
    results.append(check_endpoint("POST", "/api/export", data=export_config, 
                                description="Export discovered data to JSON"))
    
    # Test 9: Stop discovery
    results.append(check_endpoint("POST", "/api/stop", description="Stop discovery and cleanup"))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("✅ All tests passed!")
        print("\nYour OpenCITE application is working correctly!")
        print(f"\nNext steps:")
        print(f"1. Open {BASE_URL} in your browser")
        print(f"2. Configure plugins through the web interface")
        print(f"3. Send OTLP traces to the endpoint shown in the status")
        print(f"4. View discovered assets in real-time")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        print("\nCheck the errors above for details.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    # Check if custom port provided
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            BASE_URL = f"http://127.0.0.1:{port}"
            print(f"Using custom port: {port}")
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}")
            sys.exit(1)
    
    sys.exit(main())

