from parser import parse_input
from validator import validate

def process(data):
    parsed = parse_input(data)
    if validate(parsed):
        return parsed["value"]
    return None
