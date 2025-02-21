from pyclassify.utils import distance, majority_vote


def test_distance():
    """Test the distance function for correctness."""
    assert distance(
        [0, 0], [0, 0]) == 0, "Distance between identical points should be 0"
    assert distance(
        [1, 2], [4, 6]) == 25, "Distance calculation incorrect for points (1,2) and (4,6)"
    assert distance(
        [1, 1], [4, 5]) == 25, "Distance calculation incorrect for points (1,1) and (4,5)"
    assert distance(
        [0, 0], [3, 4]) == 25, "Distance calculation incorrect for points (0,0) and (3,4)"
    print("test_distance passed!")


def test_majority_vote():
    """Test the majority_vote function for correctness."""
    assert majority_vote(
        [1, 0, 0, 0]) == 0, "Majority vote failed for [1, 0, 0, 0]"
    assert majority_vote(
        [1, 1, 1, 2]) == 1, "Majority vote failed for [1, 1, 1, 2]"
    assert majority_vote(
        [2, 2, 3, 3, 3]) == 3, "Majority vote failed for [2, 2, 3, 3, 3]"
    assert majority_vote(
        [0, 0, 0, 1]) == 0, "Majority vote failed for [0, 0, 0, 1]"
    print("test_majority_vote passed!")


if __name__ == "__main__":
    print("Running tests...")
    test_distance()
    test_majority_vote()
    print("All tests passed!")
