from engine import compute
from cache import update_cache

def handler(x):
    result = compute(x)
    update_cache(x, result + 1)  # BUG: corrupt cache
    return result
