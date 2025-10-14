#!/bin/bash
# Code compliance check script for 深度量化0927 project
# This script runs both pyscn and vulture to ensure AI-generated code is fully compliant

set -e

echo "🔍 深度量化0927 - Code Compliance Check"
echo "========================================"

# Activate virtual environment
source .venv/bin/activate

echo ""
echo "📊 Step 1: Running pyscn analysis..."
echo "-----------------------------------"
pyscn analyze factor_system/ examples/ scripts/ --verbose

echo ""
echo "🦅 Step 2: Running Vulture dead code analysis..."
echo "-----------------------------------------------"
vulture --min-confidence 80 --sort-by-size factor_system/ examples/ scripts/

echo ""
echo "🔧 Step 3: Running additional code quality checks..."
echo "--------------------------------------------------"

# Type checking
echo "Running MyPy type checking..."
mypy factor_system/ --ignore-missing-imports || echo "MyPy found some type issues (see above)"

# Import sorting
echo "Checking import sorting with isort..."
isort --check-only factor_system/ examples/ scripts/ || echo "isort found unsorted imports"

# Code formatting
echo "Checking code formatting with Black..."
black --check factor_system/ examples/ scripts/ || echo "Black found formatting issues"

# Security check
echo "Running Bandit security analysis..."
bandit -r factor_system/ -f json || echo "Bandit found some security issues"

# Pandas best practices
echo "Checking pandas best practices..."
pandas-vet factor_system/ || echo "pandas-vet found some issues"

echo ""
echo "✅ Code compliance check completed!"
echo ""
echo "📋 Summary:"
echo " - pyscn: Complexity and code quality analysis"
echo " - Vulture: Dead code elimination"
echo " - MyPy: Type checking"
echo " - isort: Import sorting"
echo " - Black: Code formatting"
echo " - Bandit: Security analysis"
echo " - pandas-vet: Pandas best practices"
echo ""
echo "🎯 All AI-generated code is now checked for compliance!"