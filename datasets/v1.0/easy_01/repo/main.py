from utils import calculate_discount

def checkout(price):
    return calculate_discount(price)

if __name__ == "__main__":
    print(checkout(100))
