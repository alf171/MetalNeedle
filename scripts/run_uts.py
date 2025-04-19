import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

if __name__ == "__main__":
    # Print debug info
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"sys.path: {sys.path}")

    # Explicitly load the test suite
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tst', pattern="test_*.py")

    # Print test info
    test_count = test_suite.countTestCases()
    print(f"Found {test_count} tests")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Exit with appropriate code
    sys.exit(not result.wasSuccessful())