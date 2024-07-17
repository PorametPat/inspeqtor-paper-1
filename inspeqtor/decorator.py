def deprecated(func):
    # Decorator to mark that function is wrong and should not be used
    def wrapper(*args, **kwargs):
        print(f"Function {func.__name__} is deprecated and should not be used.")
        return func(*args, **kwargs)

    return wrapper


def not_yet_tested(func):
    # Decorator to that function is not yet tested
    def wrapper(*args, **kwargs):
        print(f"Function {func.__name__} is not yet tested.")
        return func(*args, **kwargs)

    return wrapper
