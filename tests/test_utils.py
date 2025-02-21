from pyclassify.utils import distance, majority_vote


def test_distance():
    assert distance([1, 2], [4, 6]) == 25  # (1-4)^2 + (2-6)^2 = 9 + 16 = 25
    assert distance([0, 0], [0, 0]) == 0
    assert distance([1, 1], [4, 5]) == 25


def test_majority_vote():
    assert majority_vote([1, 2, 2, 3, 2]) == 2
    assert majority_vote([1, 1, 1, 2]) == 1
    assert majority_vote([2, 2, 3, 3, 3]) == 3


if __name__ == "__main__":
    print("Running distance tests...")
    test_distance()
    print("✅ distance tests passed!")

    print("Running majority_vote tests...")
    test_majority_vote()
    print("✅ majority_vote tests passed!")
