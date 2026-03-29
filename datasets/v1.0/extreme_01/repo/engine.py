from cache import get_cache

def compute(x):
    cache = get_cache()
    if x in cache:
        return cache[x]
    return x * 2
