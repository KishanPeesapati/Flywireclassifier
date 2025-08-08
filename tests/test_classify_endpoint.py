# Test data
VALID_TEST_CASES = [
        {
            "query": "My card was declined at the store", 
            "expected_intent": "declined_card_payment"
        },
        {
            "query": "I cannot activate my new credit card", 
            "expected_intent": "activate_my_card"
        },
        {
            "query": "When will my replacement card arrive", 
            "expected_intent": "card_delivery"
        },
        {
            "query": "There is a wrong charge on my statement", 
            "expected_intent": "card_payment_fee_charged"
        }
    ]
    
INVALID_TEST_CASES = [
        "1222",           # Numbers only
        "",               # Empty string
        "a",              # Too short
        "!!!",            # No letters
        "   ",            # Whitespace only
    ]
"""
Tests for the /classify endpoint - core service functionality.

These tests validate the main classification endpoint for critical functionality
including input validation, confidence thresholding, error handling, and 
business logic correctness.
"""

import requests
import time
import json
from typing import Dict, List


class TestClassifyEndpoint:
    """Test suite for /classify endpoint."""
    
    # Configuration - can be overridden for Docker testing
    BASE_URL = "http://localhost:8000"
    CLASSIFY_URL = f"{BASE_URL}/classify"
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize test suite with configurable base URL."""
        self.BASE_URL = base_url
        self.CLASSIFY_URL = f"{base_url}/classify"
        self.results = {
            "happy_path": [],
            "input_validation": [],
            "confidence_threshold": [],
            "error_handling": [],
            "business_logic": []
        }
        
        # Test data
        self.VALID_TEST_CASES = [
            {
                "query": "My card was declined at the store", 
                "expected_intent": "declined_card_payment"
            },
            {
                "query": "I cannot activate my new credit card", 
                "expected_intent": "activate_my_card"
            },
            {
                "query": "When will my replacement card arrive", 
                "expected_intent": "card_delivery"
            },
            {
                "query": "There is a wrong charge on my statement", 
                "expected_intent": "card_payment_fee_charged"
            }
        ]
        
        self.INVALID_TEST_CASES = [
            "1222",           # Numbers only
            "",               # Empty string
            "a",              # Too short
            "!!!",            # No letters
            "   ",            # Whitespace only
        ]
    
    def test_service_availability(self) -> bool:
        """Test if the service is running and accessible."""
        try:
            response = requests.get(f"{self.BASE_URL}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def test_happy_path(self) -> Dict:
        """Test valid queries return proper classifications."""
        print("Testing happy path scenarios...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        for test_case in self.VALID_TEST_CASES:
            try:
                start_time = time.time()
                response = requests.post(
                    self.CLASSIFY_URL,
                    json={"query": test_case["query"]},
                    timeout=10
                )
                processing_time = time.time() - start_time
                
                # Check response status
                if response.status_code != 200:
                    results["failed"] += 1
                    results["errors"].append(f"Query '{test_case['query']}' returned status {response.status_code}")
                    continue
                
                # Check response format
                data = response.json()
                required_fields = ["intent_label", "intent_description", "confidence", "processing_time_ms"]
                
                if not all(field in data for field in required_fields):
                    results["failed"] += 1
                    results["errors"].append(f"Query '{test_case['query']}' missing required fields")
                    continue
                
                # Check processing time
                if processing_time > 2.0:
                    results["failed"] += 1
                    results["errors"].append(f"Query '{test_case['query']}' took {processing_time:.2f}s (too slow)")
                    continue
                
                # Check confidence range
                confidence = data.get("confidence", 0)
                if not (0.0 <= confidence <= 1.0):
                    results["failed"] += 1
                    results["errors"].append(f"Query '{test_case['query']}' confidence {confidence} out of range")
                    continue
                
                results["passed"] += 1
                print(f"  PASS: '{test_case['query']}' -> {data['intent_description']} (confidence: {confidence:.3f})")
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Query '{test_case['query']}' caused exception: {str(e)}")
        
        return results
    
    def test_input_validation(self) -> Dict:
        """Test invalid inputs are properly rejected."""
        print("Testing input validation...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        for invalid_query in self.INVALID_TEST_CASES:
            try:
                response = requests.post(
                    self.CLASSIFY_URL,
                    json={"query": invalid_query},
                    timeout=5
                )
                
                # Should return 400 for invalid input
                if response.status_code in [400, 422]:
                    results["passed"] += 1
                    print(f"  PASS: '{invalid_query}' correctly rejected with {response.status_code}")
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Invalid query '{invalid_query}' not rejected (status: {response.status_code})")
                    
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Invalid query '{invalid_query}' caused exception: {str(e)}")
        
        # Test malformed JSON
        try:
            response = requests.post(
                self.CLASSIFY_URL,
                data="invalid json",
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 400 or response.status_code == 422:
                results["passed"] += 1
                print("  PASS: Malformed JSON correctly rejected")
            else:
                results["failed"] += 1
                results["errors"].append(f"Malformed JSON not rejected (status: {response.status_code})")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Malformed JSON test caused exception: {str(e)}")
        
        return results
    
    def test_confidence_threshold(self) -> Dict:
        """Test confidence threshold behavior."""
        print("Testing confidence threshold behavior...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        # Test with a query that should have high confidence
        high_confidence_query = "My credit card payment was declined"
        
        try:
            response = requests.post(
                self.CLASSIFY_URL,
                json={"query": high_confidence_query},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                confidence = data.get("confidence", 0)
                
                if confidence >= 0.6:
                    # Should get normal response
                    if "is_confident" in data and data["is_confident"]:
                        results["passed"] += 1
                        print(f"  PASS: High confidence query handled correctly (confidence: {confidence:.3f})")
                    else:
                        results["failed"] += 1
                        results["errors"].append("High confidence query not marked as confident")
                else:
                    # Should get low confidence response
                    if "message" in data or not data.get("is_confident", True):
                        results["passed"] += 1
                        print(f"  PASS: Low confidence handled correctly (confidence: {confidence:.3f})")
                    else:
                        results["failed"] += 1
                        results["errors"].append("Low confidence query not handled properly")
            else:
                results["failed"] += 1
                results["errors"].append(f"Confidence test query failed with status {response.status_code}")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Confidence threshold test caused exception: {str(e)}")
        
        return results
    
    def test_error_handling(self) -> Dict:
        """Test error handling scenarios."""
        print("Testing error handling...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        # Test missing query field
        try:
            response = requests.post(
                self.CLASSIFY_URL,
                json={"not_query": "test"},
                timeout=5
            )
            
            if response.status_code == 422:  # Pydantic validation error
                results["passed"] += 1
                print("  PASS: Missing query field correctly rejected")
            else:
                results["failed"] += 1
                results["errors"].append(f"Missing query field not rejected (status: {response.status_code})")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Missing field test caused exception: {str(e)}")
        
        # Test wrong HTTP method
        try:
            response = requests.get(self.CLASSIFY_URL, timeout=5)
            
            if response.status_code == 405:  # Method not allowed
                results["passed"] += 1
                print("  PASS: Wrong HTTP method correctly rejected")
            else:
                results["failed"] += 1
                results["errors"].append(f"Wrong HTTP method not rejected (status: {response.status_code})")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Wrong method test caused exception: {str(e)}")
        
        return results
    
    def test_business_logic(self) -> Dict:
        """Test business logic correctness."""
        print("Testing business logic...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        # Test consistency - same query should return same result
        test_query = "My card was declined"
        
        try:
            responses = []
            for i in range(3):
                response = requests.post(
                    self.CLASSIFY_URL,
                    json={"query": test_query},
                    timeout=5
                )
                if response.status_code == 200:
                    responses.append(response.json())
            
            if len(responses) == 3:
                # Check if all responses have same intent_label
                intent_labels = [r.get("intent_label") for r in responses]
                if len(set(intent_labels)) == 1:
                    results["passed"] += 1
                    print(f"  PASS: Consistent results for repeated queries")
                else:
                    results["failed"] += 1
                    results["errors"].append("Inconsistent results for same query")
            else:
                results["failed"] += 1
                results["errors"].append("Failed to get responses for consistency test")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Consistency test caused exception: {str(e)}")
        
        return results
    
    def run_all_tests(self) -> Dict:
        """Run all test categories and return summary."""
        print("Running /classify endpoint tests...\n")
        
        # Check if service is available
        if not self.test_service_availability():
            print("ERROR: Service not available at http://localhost:8000")
            print("Please start the service with: python main.py")
            return {"error": "Service not available"}
        
        print("Service is available. Starting tests...\n")
        
        # Run all test categories
        self.results["happy_path"] = self.test_happy_path()
        print()
        self.results["input_validation"] = self.test_input_validation()
        print()
        self.results["confidence_threshold"] = self.test_confidence_threshold()
        print()
        self.results["error_handling"] = self.test_error_handling()
        print()
        self.results["business_logic"] = self.test_business_logic()
        print()
        
        return self.results
    
    def print_summary(self, results: Dict):
        """Print test summary."""
        print("=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        
        total_passed = 0
        total_failed = 0
        
        for category, result in results.items():
            if isinstance(result, dict) and "passed" in result:
                passed = result["passed"]
                failed = result["failed"]
                total_passed += passed
                total_failed += failed
                
                status = "PASS" if failed == 0 else "FAIL"
                print(f"{category.replace('_', ' ').title()}: {passed}/{passed + failed} passed [{status}]")
                
                # Print errors if any
                if result["errors"]:
                    for error in result["errors"]:
                        print(f"  ERROR: {error}")
        
        print(f"\nOverall: {total_passed}/{total_passed + total_failed} tests passed")
        
        if total_failed == 0:
            print("\nAll tests passed! Service is working correctly.")
        else:
            print(f"\n{total_failed} tests failed. Please review errors above.")


if __name__ == "__main__":
    # Run tests if called directly
    test_suite = TestClassifyEndpoint()
    results = test_suite.run_all_tests()
    test_suite.print_summary(results)