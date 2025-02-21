
import os
import argparse
from typing import List, Any
from pyclassify.utils import read_file, read_config
from pyclassify.classifier import kNN


def split_dataset(features: List[List[float]], labels: List[Any], test_split: float):
    """
    Splits the dataset into training and testing sets.

    Args:
        features (List[List[float]]): Feature matrix.
        labels (List[Any]): Labels corresponding to features.
        test_split (float): Fraction of data to use as test set.

    Returns:
        Tuple: train_features, train_labels, test_features, test_labels
    """
    split_index = int(len(features) * (1 - test_split))
    train_features = features[:split_index]
    train_labels = labels[:split_index]
    test_features = features[split_index:]
    test_labels = labels[split_index:]
    return train_features, train_labels, test_features, test_labels


def compute_accuracy(predictions: List[Any], true_labels: List[Any]) -> float:
    """
    Computes accuracy of predictions.

    Args:
        predictions (List[Any]): Predicted labels.
        true_labels (List[Any]): Actual labels.

    Returns:
        float: Accuracy percentage.
    """
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    return (correct / len(true_labels)) * 100


def main():
    parser = argparse.ArgumentParser(description="Run kNN classification.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    args = parser.parse_args()

    # Read configuration
    config = read_config(args.config)
    k = config['k']
    dataset_path = config['dataset_path']
    test_split = config['test_split']

    # Load dataset
    features, labels = read_file(dataset_path)

    # Split dataset
    train_features, train_labels, test_features, test_labels = split_dataset(
        features, labels, test_split)

    # Initialize and run kNN
    classifier = kNN(k=k)
    predictions = classifier((train_features, train_labels), test_features)

    # Compute and print accuracy
    accuracy = compute_accuracy(predictions, test_labels)
    print(f"✅ Classification Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
