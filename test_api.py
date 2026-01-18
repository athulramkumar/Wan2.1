#!/usr/bin/env python3
"""
Wan2.1 API Tests

Tests for the video generation API.

Usage:
    # Unit tests (no server needed)
    python test_api.py --unit
    
    # Integration tests (server must be running)
    python test_api.py --integration --server http://localhost:30396
    
    # All tests
    python test_api.py --all --server http://localhost:30396
"""

import argparse
import json
import os
import sys
import time
import unittest
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Unit Tests (No Server Required)
# =============================================================================

class TestCachingUtils(unittest.TestCase):
    """Test the caching utility functions."""
    
    def setUp(self):
        from api.utils.caching import should_use_cache, get_cache_statistics
        self.should_use_cache = should_use_cache
        self.get_cache_statistics = get_cache_statistics
    
    def test_before_start_step_no_cache(self):
        """Steps before start_step should not use cache."""
        # With start_step=10, steps 0-9 should return False
        for step in range(10):
            result = self.should_use_cache(step, start_step=10, end_step=40, interval=3)
            self.assertFalse(result, f"Step {step} should NOT use cache (before start)")
    
    def test_after_end_step_no_cache(self):
        """Steps after end_step should not use cache."""
        # With end_step=40, steps 41+ should return False
        for step in range(41, 50):
            result = self.should_use_cache(step, start_step=10, end_step=40, interval=3)
            self.assertFalse(result, f"Step {step} should NOT use cache (after end)")
    
    def test_interval_boundary_no_cache(self):
        """On interval boundaries, should compute fresh (not use cache)."""
        # With start_step=10, interval=3:
        # Fresh steps (0 mod 3): 10, 13, 16, 19, 22, ...
        fresh_steps = [10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40]
        for step in fresh_steps:
            result = self.should_use_cache(step, start_step=10, end_step=40, interval=3)
            self.assertFalse(result, f"Step {step} should be FRESH (interval boundary)")
    
    def test_between_intervals_use_cache(self):
        """Between interval boundaries, should use cache."""
        # With start_step=10, interval=3:
        # Cache steps (non-0 mod 3): 11, 12, 14, 15, 17, 18, ...
        cache_steps = [11, 12, 14, 15, 17, 18, 20, 21]
        for step in cache_steps:
            result = self.should_use_cache(step, start_step=10, end_step=40, interval=3)
            self.assertTrue(result, f"Step {step} should USE CACHE")
    
    def test_none_end_step(self):
        """When end_step is None, cache until the end."""
        # Step 45 with end_step=None should still be in caching zone
        result = self.should_use_cache(45, start_step=10, end_step=None, interval=3)
        # 45 - 10 = 35, 35 % 3 = 2 != 0, so should use cache
        self.assertTrue(result, "Step 45 should use cache when end_step=None")
    
    def test_cache_statistics(self):
        """Test cache statistics calculation."""
        stats = self.get_cache_statistics(
            total_steps=50,
            start_step=10,
            end_step=40,
            interval=3
        )
        
        self.assertEqual(stats["total_steps"], 50)
        self.assertGreater(stats["cache_hits"], 0)
        self.assertGreater(stats["fresh_computes"], 0)
        self.assertEqual(stats["cache_hits"] + stats["fresh_computes"], 50)
        self.assertGreater(stats["cache_hit_rate"], 0)
        self.assertLess(stats["cache_hit_rate"], 1)
        self.assertGreater(stats["estimated_speedup"], 1.0)
        
        print(f"\nCache statistics: {stats}")


class TestSchedulingUtils(unittest.TestCase):
    """Test the scheduling utility functions."""
    
    def setUp(self):
        from api.utils.scheduling import (
            validate_schedule,
            parse_schedule,
            get_model_for_step,
            get_default_schedule,
        )
        self.validate_schedule = validate_schedule
        self.parse_schedule = parse_schedule
        self.get_model_for_step = get_model_for_step
        self.get_default_schedule = get_default_schedule
    
    def test_valid_schedule(self):
        """Valid schedule should pass validation."""
        schedule = [["14B", 15], ["1.3B", 35]]
        is_valid, error = self.validate_schedule(schedule, total_steps=50)
        self.assertTrue(is_valid, f"Should be valid: {error}")
    
    def test_invalid_schedule_wrong_sum(self):
        """Schedule with wrong sum should fail."""
        schedule = [["14B", 10], ["1.3B", 30]]  # Sum = 40, not 50
        is_valid, error = self.validate_schedule(schedule, total_steps=50)
        self.assertFalse(is_valid)
        self.assertIn("40", error)
        self.assertIn("50", error)
    
    def test_invalid_schedule_bad_model(self):
        """Schedule with invalid model name should fail."""
        schedule = [["14B", 15], ["2B", 35]]  # "2B" is not valid
        is_valid, error = self.validate_schedule(schedule, total_steps=50)
        self.assertFalse(is_valid)
        self.assertIn("2B", error)
    
    def test_parse_schedule(self):
        """Test schedule parsing from JSON format."""
        schedule = [["14B", 15], ["1.3B", 35]]
        parsed = self.parse_schedule(schedule)
        
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0], ("14B", 15))
        self.assertEqual(parsed[1], ("1.3B", 35))
    
    def test_get_model_for_step(self):
        """Test step-to-model mapping."""
        schedule = [("14B", 15), ("1.3B", 35)]
        
        # Steps 0-14 should be 14B
        for step in range(15):
            model = self.get_model_for_step(step, schedule)
            self.assertEqual(model, "14B", f"Step {step} should use 14B")
        
        # Steps 15-49 should be 1.3B
        for step in range(15, 50):
            model = self.get_model_for_step(step, schedule)
            self.assertEqual(model, "1.3B", f"Step {step} should use 1.3B")
    
    def test_default_schedule(self):
        """Test default schedule generation."""
        schedule = self.get_default_schedule(50)
        
        total = sum(steps for _, steps in schedule)
        self.assertEqual(total, 50)
        
        # Should have both models
        models = {m for m, _ in schedule}
        self.assertIn("14B", models)
        self.assertIn("1.3B", models)
        
        print(f"\nDefault schedule for 50 steps: {schedule}")


# =============================================================================
# Integration Tests (Server Required)
# =============================================================================

class TestAPIIntegration(unittest.TestCase):
    """Integration tests that require a running server."""
    
    server_url: str = "http://localhost:30396"
    
    @classmethod
    def setUpClass(cls):
        """Check if server is running."""
        import requests
        try:
            response = requests.get(f"{cls.server_url}/health", timeout=5)
            if response.status_code != 200:
                raise unittest.SkipTest("Server not healthy")
        except Exception as e:
            raise unittest.SkipTest(f"Server not available: {e}")
    
    def test_health_endpoint(self):
        """Test /health endpoint."""
        import requests
        
        response = requests.get(f"{self.server_url}/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "Server is healthy")
        self.assertIn("14B", data["models_loaded"])
        self.assertIn("1.3B", data["models_loaded"])
        self.assertIn("gpu_memory_used", data)
        
        print(f"\nHealth: {data}")
    
    def test_config_endpoint(self):
        """Test /config endpoint."""
        import requests
        
        response = requests.get(f"{self.server_url}/config")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("defaults", data)
        self.assertIn("valid_models", data)
        self.assertIn("valid_frame_counts", data)
        
        print(f"\nDefaults: {data['defaults']}")
    
    def test_generate_invalid_request(self):
        """Test validation of invalid generation request."""
        import requests
        
        # Missing prompt
        response = requests.post(
            f"{self.server_url}/generate",
            json={"model": "baseline_14B"}
        )
        self.assertEqual(response.status_code, 422)  # Validation error
    
    def test_job_not_found(self):
        """Test 404 for non-existent job."""
        import requests
        
        response = requests.get(f"{self.server_url}/status/nonexistent123")
        self.assertEqual(response.status_code, 404)
    
    def test_full_generation_cycle_1_3B(self):
        """Test complete generation cycle with 1.3B model (fastest)."""
        import requests
        
        print("\n" + "=" * 60)
        print("INTEGRATION TEST: Full Generation Cycle (1.3B)")
        print("=" * 60)
        
        # Submit job
        response = requests.post(
            f"{self.server_url}/generate",
            json={
                "prompt": "A simple test video with moving shapes",
                "model": "baseline_1.3B",
                "frame_count": 17,  # Minimum frames for speed
                "sampling_steps": 30,  # Fewer steps for speed
            }
        )
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        job_id = data["job_id"]
        print(f"Job submitted: {job_id}")
        
        # Poll for completion
        max_wait = 300  # 5 minutes max
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = requests.get(f"{self.server_url}/status/{job_id}")
            self.assertEqual(response.status_code, 200)
            
            status_data = response.json()
            print(f"  Status: {status_data['status']}, Progress: {status_data['progress']}%")
            
            if status_data["status"] == "completed":
                print(f"\n✓ Generation completed!")
                print(f"  Time: {status_data.get('generation_time', 'N/A')}s")
                print(f"  Video URL: {status_data.get('video_url', 'N/A')}")
                
                # Try to download video
                video_response = requests.get(f"{self.server_url}/video/{job_id}")
                self.assertEqual(video_response.status_code, 200)
                self.assertIn("video/mp4", video_response.headers.get("content-type", ""))
                print(f"  Video size: {len(video_response.content)} bytes")
                
                return
            
            elif status_data["status"] == "failed":
                self.fail(f"Generation failed: {status_data.get('error', 'Unknown error')}")
            
            time.sleep(5)
        
        self.fail("Generation timed out")


# =============================================================================
# Test Runner
# =============================================================================

def run_unit_tests():
    """Run unit tests only."""
    print("\n" + "=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestCachingUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestSchedulingUtils))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests(server_url: str):
    """Run integration tests."""
    print("\n" + "=" * 60)
    print("RUNNING INTEGRATION TESTS")
    print(f"Server: {server_url}")
    print("=" * 60 + "\n")
    
    # Set server URL
    TestAPIIntegration.server_url = server_url
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAPIIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description="Wan2.1 API Tests")
    
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only"
    )
    
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:30396",
        help="Server URL for integration tests"
    )
    
    args = parser.parse_args()
    
    # Default to unit tests if nothing specified
    if not any([args.unit, args.integration, args.all]):
        args.unit = True
    
    success = True
    
    if args.unit or args.all:
        success = run_unit_tests() and success
    
    if args.integration or args.all:
        success = run_integration_tests(args.server) and success
    
    print("\n" + "=" * 60)
    if success:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

