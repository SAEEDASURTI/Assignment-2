import numpy as np

def rmse(predictions, targets):
    pred = np.array(predictions)
    tar = np.array(targets)
    mse = np.mean((pred - tar) ** 2)
    return np.sqrt(mse)

# Example usage
if __name__ == "__main__":
    predictions = [1, 2, 3, 4, 5]
    targets = [1, 2, 3, 4, 5]
    print(f"RMSE: {rmse(predictions, targets)}")
