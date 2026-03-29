_cache = {}

def get_cache():
    return _cache

def update_cache(key, value):
    _cache[key] = value
