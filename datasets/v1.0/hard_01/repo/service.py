from db import fetch_user

def get_user_score(name):
    user = fetch_user(name)
    return user["points"] * 2  # BUG: wrong key ("points")
