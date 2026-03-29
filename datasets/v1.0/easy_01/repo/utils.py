def calculate_discount(price):
    discount = 0.1
    return price - (price * discount * 10)  # BUG: extra *10
