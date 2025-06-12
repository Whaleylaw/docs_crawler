#!/usr/bin/env python3
"""
Test runner for Crawl4AI Standalone Application.

This script provides various test execution modes:
- Unit tests only (fast)
- Integration tests
- API tests
- Full test suite
- Coverage reporting
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(f"⚠️  Warnings/Errors:\n{result.stderr}")
    
    if result.returncode != 0:
        print(f"❌ {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ {description} completed successfully")
        return True


def setup_environment():
    """Setup test environment variables."""
    print("🔧 Setting up test environment...")
    
    # Set test environment variables
    test_env = {
        "ENVIRONMENT": "test",
        "TEST_MODE": "true",
        "SUPABASE_URL": os.getenv("TEST_SUPABASE_URL", "http://localhost:8000"),
        "SUPABASE_SERVICE_KEY": os.getenv("TEST_SUPABASE_KEY", "test-key"),
        "OPENAI_API_KEY": os.getenv("TEST_OPENAI_API_KEY", "test-key"),
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
        print(f"  {key}={value}")


def check_dependencies():
    """Check if test dependencies are installed."""
    print("📦 Checking test dependencies...")
    
    required_packages = [
        "pytest",
        "pytest-asyncio", 
        "pytest-cov",
        "pytest-mock",
        "httpx",  # For FastAPI test client
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ❌ {package}")
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install pytest pytest-asyncio pytest-cov pytest-mock httpx")
        return False
    
    return True


def run_unit_tests(verbose=False):
    """Run unit tests only."""
    cmd = ["python", "-m", "pytest", "tests/unit/", "-m", "unit"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Running unit tests")


def run_integration_tests(verbose=False):
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/", "-m", "integration"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Running integration tests")


def run_api_tests(verbose=False):
    """Run API tests."""
    cmd = ["python", "-m", "pytest", "tests/api/", "-m", "api"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Running API tests")


def run_all_tests(verbose=False, coverage=True):
    """Run all tests with optional coverage."""
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=components",
            "--cov-report=term-missing",
            "--cov-report=html:tests/coverage_html"
        ])
    
    return run_command(cmd, "Running full test suite")


def run_specific_test(test_path, verbose=False):
    """Run a specific test file or test function."""
    cmd = ["python", "-m", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, f"Running specific test: {test_path}")


def generate_coverage_report():
    """Generate detailed coverage report."""
    print("\n📊 Generating coverage report...")
    
    # HTML report
    cmd = ["python", "-m", "pytest", "--cov=components", "--cov-report=html:tests/coverage_html", "tests/"]
    if run_command(cmd, "Generating HTML coverage report"):
        print(f"📄 HTML coverage report: {Path('tests/coverage_html/index.html').absolute()}")
    
    # Terminal report
    cmd = ["python", "-m", "pytest", "--cov=components", "--cov-report=term", "tests/"]
    run_command(cmd, "Generating terminal coverage report")


def lint_code():
    """Run code linting."""
    print("\n🔍 Running code linting...")
    
    # Check if flake8 is available
    try:
        import flake8
        cmd = ["python", "-m", "flake8", "components/", "tests/", "--max-line-length=100"]
        run_command(cmd, "Running flake8 linting")
    except ImportError:
        print("⚠️  flake8 not installed, skipping linting")
    
    # Check if black is available
    try:
        import black
        cmd = ["python", "-m", "black", "--check", "components/", "tests/"]
        run_command(cmd, "Checking code formatting with black")
    except ImportError:
        print("⚠️  black not installed, skipping format checking")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Test runner for Crawl4AI Standalone Application")
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "api", "all", "coverage", "lint", "specific"],
        help="Type of tests to run"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--test-path", help="Specific test path (for 'specific' type)")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    parser.add_argument("--setup-only", action="store_true", help="Only setup environment and check dependencies")
    
    args = parser.parse_args()
    
    print("🧪 Crawl4AI Test Runner")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    if args.setup_only:
        print("✅ Environment setup complete!")
        sys.exit(0)
    
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Run tests based on type
    success = True
    
    if args.test_type == "unit":
        success = run_unit_tests(args.verbose)
    elif args.test_type == "integration":
        success = run_integration_tests(args.verbose)
    elif args.test_type == "api":
        success = run_api_tests(args.verbose)
    elif args.test_type == "all":
        success = run_all_tests(args.verbose, not args.no_coverage)
    elif args.test_type == "coverage":
        success = generate_coverage_report()
    elif args.test_type == "lint":
        success = lint_code()
    elif args.test_type == "specific":
        if not args.test_path:
            print("❌ --test-path required for 'specific' test type")
            sys.exit(1)
        success = run_specific_test(args.test_path, args.verbose)
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()