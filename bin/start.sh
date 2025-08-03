#!/bin/bash

# Memory optimization startup script
echo "🚀 Starting Audio Alert API with memory optimizations..."

# Set memory limits and optimizations
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=100000
export MALLOC_MMAP_MAX_=65536

# Disable memory-hungry features
export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=/tmp
export LIBROSA_CACHE_DIR=/tmp
export SCIPY_PIL_IMAGE_VIEWER=disabled

# TensorFlow optimizations
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_DISABLE_MKL=1

# Threading limits
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Python optimizations
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=1

# Create temp directories if they don't exist
mkdir -p /tmp/numba_cache
mkdir -p /tmp/librosa_cache

# Precompile critical modules to avoid runtime compilation
echo "📦 Precompiling critical modules..."
python3 -c "
import warnings
warnings.filterwarnings('ignore')

print('Loading numpy...')
import numpy as np

print('Loading joblib...')
import joblib

print('Loading basic scipy...')
try:
    from scipy import signal
except:
    pass

print('Precompilation complete!')
" 2>/dev/null

# Start the application with memory monitoring
echo "🌐 Starting Flask application..."
exec gunicorn \
    --bind 0.0.0.0:${PORT:-5000} \
    --workers 1 \
    --worker-class sync \
    --worker-connections 100 \
    --max-requests 50 \
    --max-requests-jitter 10 \
    --timeout 120 \
    --keep-alive 2 \
    --preload \
    main:app