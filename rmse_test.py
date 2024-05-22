import rmse

def test_rmse():
    predictions = [1, 2, 3, 4, 5]
    targets = [1, 2, 3, 4, 5]
    assert rmse.rmse(predictions, targets) == 0

    predictions = [2, 3, 4, 5, 6]
    targets = [1, 2, 3, 4, 5]
    assert rmse.rmse(predictions, targets) == 1

    predictions = [0, 0, 0, 0, 0]
    targets = [1, 1, 1, 1, 1]
    assert rmse.rmse(predictions, targets) == 1

    predictions = [1.5, 2.5, 3.5, 4.5, 5.5]
    targets = [1, 2, 3, 4, 5]
    assert rmse.rmse(predictions, targets) == 0.5

if __name__ == "__main__":
    test_rmse()
    print("All tests passed!")
