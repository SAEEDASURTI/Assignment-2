import torch

def create_tensor_of_val(dimensions, val):
    """
    Create a tensor of the given dimensions, filled with the value of `val`.
    dimensions is a tuple of integers.
    """
    res = torch.ones(dimensions) * val  # Create a tensor of ones and multiply by val
    return res

def calculate_elementwise_product(A, B):
    """
    Calculate the elementwise product of the two tensors A and B.
    Note that the dimensions of A and B should be the same.
    """
    res = A * B  # Element-wise multiplication of tensors
    return res

def calculate_matrix_product(X, W):
    """
    Calculate the product of the two tensors X and W.
    Note that the dimensions of X and W should be compatible for multiplication.
    """
    res = torch.matmul(X, W.T)  # Matrix multiplication
    return res

def calculate_matrix_prod_with_bias(X, W, b):
    """
    Calculate the product of the two tensors X and W and add the bias.
    Note that the dimensions of X and W should be compatible for multiplication.
    """
    res = torch.matmul(X, W.T) + b  # Matrix multiplication and add bias
    return res

def calculate_activation(sum_total):
    """
    Calculate a step function as an activation of the neuron.
    """
    res = torch.heaviside(sum_total, torch.tensor(0.0))  # Step function activation
    return res

def calculate_output(X, W, b):
    """
    Calculate the output of the neuron.
    """
    sum_total = calculate_matrix_prod_with_bias(X, W, b)  # Calculate sum with bias
    res = calculate_activation(sum_total)  # Apply activation function
    return res
