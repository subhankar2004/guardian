#!/bin/bash

# Setup script to make all scripts executable and prepare deployment
echo "🔧 Setting up Audio Alert API for deployment..."

# Create bin directory if it doesn't exist
mkdir -p bin

# Make scripts executable
chmod +x bin/start.sh
chmod +x setup.sh

echo "✅ Scripts are now executable"

# Test if all required files exist
echo "📋 Checking required files..."

required_files=("main.py" "requirements.txt" "svm_model.pkl" "mlp_model.h5" "bin/start.sh")

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file - Found"
    else
        echo "❌ $file - Missing"
    fi
done

echo "📁 Project structure:"
find . -type f -name "*.py" -o -name "*.sh" -o -name "*.pkl" -o -name "*.h5" -o -name "*.txt" | sort

echo "🚀 Setup complete! Ready for deployment."