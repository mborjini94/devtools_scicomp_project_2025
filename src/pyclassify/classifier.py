from typing import List, Any, Tuple
# Import majority_vote function
from pyclassify.utils import distance, majority_vote


class kNN:
    def __init__(self, k: int):
        """
        Initialize the kNN classifier.

        Args:
            k (int): Number of nearest neighbors to consider.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k

    def _get_k_nearest_neighbors(self, X: List[List[float]], y: List[Any], x: List[float]) -> List[Any]:
        """
        Get the labels of the k nearest neighbors to point x.

        Args:
            X (List[List[float]]): Dataset points.
            y (List[Any]): Labels corresponding to the dataset points.
            x (List[float]): New data point to classify.

        Returns:
            List[Any]: Labels of the k nearest neighbors.
        """
        if len(X) != len(y):
            raise ValueError(
                "The number of data points and labels must be the same.")

        distances = [(distance(point, x), label) for point, label in zip(X, y)]
        distances.sort(key=lambda pair: pair[0])
        nearest_neighbors = [label for _, label in distances[:self.k]]

        return nearest_neighbors

    def __call__(self, data: Tuple[List[List[float]], List[Any]], new_points: List[List[float]]) -> List[Any]:
        """
        Predict the class for each point in new_points.

        Args:
            data (Tuple[List[List[float]], List[Any]]): Tuple containing:
                - X (List[List[float]]): Feature matrix of the dataset.
                - y (List[Any]): Labels corresponding to the dataset points.
            new_points (List[List[float]]): New points to classify.

        Returns:
            List[Any]: Predicted classes for all new points.
        """
        X, y = data
        predictions = []

        for point in new_points:
            neighbors = self._get_k_nearest_neighbors(X, y, point)
            # Use the majority_vote function
            predicted_class = majority_vote(neighbors)
            predictions.append(predicted_class)

        return predictions
