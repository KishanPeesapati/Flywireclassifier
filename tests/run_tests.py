"""
Simple test runner for the credit card intent classification service.

This script runs all critical tests for the /classify endpoint and provides
a summary of results. Designed for quick validation during development.
"""

import sys
import os
import time
import subprocess
import requests

# Add parent directory to path to import test modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_classify_endpoint import TestClassifyEndpoint


def check_service_running(url: str = "http://localhost:8000", timeout: int = 5) -> bool:
    """Check if the service is running."""
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def wait_for_service(url: str = "http://localhost:8000", max_wait: int = 30) -> bool:
    """Wait for service to be available."""
    print(f"Waiting for service at {url}...")
    
    for i in range(max_wait):
        if check_service_running(url):
            print(f"Service is ready after {i + 1} seconds.")
            return True
        time.sleep(1)
        if i % 5 == 4:  # Print progress every 5 seconds
            print(f"Still waiting... ({i + 1}/{max_wait} seconds)")
    
    return False


def run_endpoint_tests(base_url: str = "http://localhost:8000"):
    """Run all /classify endpoint tests."""
    print("Credit Card Intent Classification Service - Test Suite")
    print("=" * 60)
    print(f"Testing service at: {base_url}")
    print()
    
    # Check if service is running
    if not check_service_running(base_url):
        print(f"Service not detected at {base_url}")
        print("Please ensure the service is running:")
        print("  Docker: docker run -p 8000:8000 credit-card-classifier")
        print("  Local: python main.py")
        print()
        return False
    
    # Initialize and run tests
    test_suite = TestClassifyEndpoint(base_url=base_url)
    start_time = time.time()
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Print summary
    test_suite.print_summary(results)
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nTest execution time: {execution_time:.1f} seconds")
    
    # Return success status
    if "error" in results:
        return False
    
    # Check if all tests passed
    total_failed = sum(r.get("failed", 0) for r in results.values() if isinstance(r, dict))
    return total_failed == 0


def main():
    """Main test runner function."""
    import sys
    
    # Check for command line arguments
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"Running tests against: {base_url}")
    success = run_endpoint_tests(base_url)
    
    if success:
        print("\nAll critical tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed. Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()