import random

def generate_random_decorator(func):
    def wrapper(*args, **kwargs):
        print("Generating random number...")
        return func(*args, **kwargs)
    return wrapper

def generate_random_number(start, end):
    return random.randint(start, end)

generate_random_number = generate_random_decorator(generate_random_number)

print((generate_random_number(1, 100)))
print((generate_random_number(1, 100)))


