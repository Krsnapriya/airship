DATABASE = {
    "alice": {"score": 50}
}

def fetch_user(name):
    return DATABASE.get(name)
