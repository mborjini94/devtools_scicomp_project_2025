import csv
import os
import yaml
from collections import Counter
from typing import List, Tuple


def distance(point1: List[float], point2: List[float]) -> float:
    """
    Calculate the square of the Euclidean distance between two points.

    Args:
        point1 (List[float]): Coordinates of the first point.
        point2 (List[float]): Coordinates of the second point.

    Returns:
        float: Square of the Euclidean distance.
    """
    if len(point1) != len(point2):
        raise ValueError(
            "Both points must have the same number of dimensions.")

    return sum((x - y) ** 2 for x, y in zip(point1, point2))


def majority_vote(neighbors: List[int]) -> int:
    """
    Return the majority class from the list of neighbors.

    Args:
        neighbors (List[int]): List of class labels.

    Returns:
        int: Majority class label.
    """
    if not neighbors:
        raise ValueError("The neighbors list cannot be empty.")

    return Counter(neighbors).most_common(1)[0][0]


def read_config(file):
    filepath = os.path.abspath(f'{file}.yaml')
    with open(filepath, 'r') as stream:
        kwargs = yaml.safe_load(stream)
    return kwargs


def read_file(filepath: str) -> Tuple[List[List[float]], List[str]]:
    """
    Reads the dataset file and returns features and labels as separate lists.

    Args:
        filepath (str): Path to the dataset file.

    Returns:
        Tuple[List[List[float]], List[str]]: Features and labels.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    features = []
    labels = []

    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 2:
                raise ValueError(f"Invalid row format: {row}")
            features.append([float(value) for value in row[:-1]])
            labels.append(row[-1])

    return features, labels
