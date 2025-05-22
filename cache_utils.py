import os
import json
import hashlib
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.analysis_cache')
CACHE_EXPIRY_DAYS = 30  # Cache results for 30 days

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def get_call_hash(call_data):
    """Generate a unique hash for a call based on its data."""
    # Create a dictionary with only the relevant fields for hashing
    hash_data = {
        'recording_url': call_data.get('Recording URL', ''),
        'call_id': call_data.get('Call ID', ''),
        'call_date': call_data.get('Call Date', ''),
        'agent': call_data.get('Agent', '')
    }
    
    # Convert to a string and generate hash
    hash_str = json.dumps(hash_data, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()

def get_cache_path(call_hash):
    """Get the cache file path for a call hash."""
    return os.path.join(CACHE_DIR, f"{call_hash}.json")

def is_cache_valid(cache_path):
    """Check if cache is still valid based on modification time."""
    if not os.path.exists(cache_path):
        return False
    
    cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    return datetime.now() - cache_time < timedelta(days=CACHE_EXPIRY_DAYS)

def save_to_cache(call_hash, result):
    """Save analysis result to cache."""
    cache_path = get_cache_path(call_hash)
    try:
        with open(cache_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'result': result
            }, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving to cache: {e}")
        return False

def load_from_cache(call_hash):
    """Load analysis result from cache if valid."""
    cache_path = get_cache_path(call_hash)
    if not is_cache_valid(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
            return data.get('result')
    except Exception as e:
        print(f"Error loading from cache: {e}")
        return None

def clear_old_cache():
    """Remove cache files older than the expiry period."""
    now = datetime.now()
    for cache_file in Path(CACHE_DIR).glob('*.json'):
        try:
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if now - file_time > timedelta(days=CACHE_EXPIRY_DAYS):
                cache_file.unlink()
        except Exception as e:
            print(f"Error clearing cache file {cache_file}: {e}")

# Clean up old cache on import
clear_old_cache()
