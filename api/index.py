"""Entry point untuk Vercel serverless function.

Vercel akan mendeteksi app Flask (WSGI) dan menjalankannya.
"""
import sys
import os
from pathlib import Path

# Pastikan root project ada di sys.path supaya `from app import app` jalan
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Vercel Python runtime akan otomatis pakai `app` sebagai WSGI handler