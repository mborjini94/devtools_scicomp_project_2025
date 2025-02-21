from pyclassify.classifier import kNN


def test_knn_call():
    # Sample dataset
    X = [[1, 2], [2, 3], [3, 4], [5, 5], [1, 0]]
    y = ['A', 'A', 'B', 'B', 'A']
    new_points = [[2, 2], [4, 4], [1, 1]]

    # Initialize kNN with k=3
    classifier = kNN(k=3)

    # Predict classes for new points
    predictions = classifier((X, y), new_points)

    print(f"Predictions for new points: {predictions}")

    assert len(predictions) == len(
        new_points), "Number of predictions should match number of new points"
    assert all(pred in set(y)
               for pred in predictions), "Predicted classes should be valid labels"


if __name__ == "__main__":
    print("Running kNN __call__ tests...")
    test_knn_call()
    print("✅ All kNN __call__ tests passed!")
