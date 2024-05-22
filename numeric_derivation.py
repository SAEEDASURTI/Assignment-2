# numeric_derivation.py

def derive(f, x, h=0.0001):
    """
    Calculate the numerical derivative of the function f at point x using central difference.
    """
    return (f(x + h) - f(x - h)) / (2 * h)
