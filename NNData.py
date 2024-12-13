"""This module contains the NNData class for managing neural network data."""

from enum import Enum
from collections import deque
import random
import numpy as np


class Order(Enum):
    """Enum for specifying the order of data."""

    RANDOM = 0
    SEQUENTIAL = 1
    SHUFFLE = 2
    STATIC = 3


class Set(Enum):
    """Enum for specifying the set of data."""

    TRAIN = 0
    TEST = 1


class NNData:
    """A class for managing neural network training and testing data."""

    @staticmethod
    def percentage_limiter(percentage: float) -> float:
        """
        Limits the given percentage to be between 0 and 1.

        :param percentage: The input percentage as a float.
        :return: A float between 0 and 1, inclusive.
        """
        return max(0.0, min(1.0, percentage))

    def __init__(self, features=None, labels=None, train_factor=0.9):
        """
        Initialize the NNData object.

        :param features: List of lists containing feature data.
        :param labels: List of lists containing label data.
        :param train_factor: Percentage of data to use for training.
        """
        self._features = None
        self._labels = None
        self._train_factor = NNData.percentage_limiter(train_factor)
        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()

        if features is not None and labels is not None:
            self.load_data(features, labels)
        else:
            self.load_data()

    def load_data(self, features=None, labels=None):
        """
        Load features and labels into the object and prepares them for use.

        :param features: List of lists containing feature data.
        :param labels: List of lists containing label data.
        """
        if features is None or labels is None:
            self._features = None
            self._labels = None
            self.split_set()
            return

        if len(features) != len(labels):
            self._features = None
            self._labels = None
            self.split_set()
            raise ValueError("Features and labels must have the same length.")

        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._features = None
            self._labels = None
            self.split_set()
            raise ValueError("Features and labels must be convertible to "
                             "float arrays.")

        self.split_set()

    def split_set(self, new_train_factor=None):
        """
        Split the data into training/testing sets based on the train_factor.

        :param new_train_factor: Optional new training factor to use.
        """
        if new_train_factor is not None:
            self._train_factor = self.percentage_limiter(new_train_factor)

        if self._features is None or self._labels is None:
            self._train_indices = []
            self._test_indices = []
            return

        num_samples = len(self._features)
        num_train = int(num_samples * self._train_factor)
        all_indices = list(range(num_samples))
        random.shuffle(all_indices)
        self._train_indices = all_indices[:num_train]
        self._test_indices = all_indices[num_train:]

    def prime_data(self, target_set=None, order=None):
        """
        Prepare the data for use by loading one or both pools.

        :param target_set: Which set to prime (Train, Test, or Both).
        :param order: Whether to shuffle the data.
        """
        if target_set is None or target_set == Set.TRAIN:
            self._train_pool = deque(self._train_indices)
            if order in (Order.RANDOM, Order.SHUFFLE):
                random.shuffle(self._train_pool)

        if target_set is None or target_set == Set.TEST:
            self._test_pool = deque(self._test_indices)
            if order in (Order.RANDOM, Order.SHUFFLE):
                random.shuffle(self._test_pool)

    def get_one_item(self, target_set=None):
        """
        Return a single item (feature and label) from the specified set.

        :param target_set: Which set to get the item from (Train or Test).
        :return: A tuple containing a feature and its corresponding label,
                 or None if the pool is empty.
        """
        if target_set is None:
            target_set = Set.TRAIN

        pool = self._train_pool if target_set == Set.TRAIN else self._test_pool

        if not pool:
            return None

        index = pool.popleft()
        return self._features[index], self._labels[index]

    def number_of_samples(self, target_set=None):
        """
        Return the number of samples in the specified set(s).

        :param target_set: Which set to count (Train, Test, or Both).
        :return: The number of samples in the specified set(s).
        """
        if target_set == Set.TRAIN:
            return len(self._train_indices)
        if target_set == Set.TEST:
            return len(self._test_indices)
        return len(self._train_indices) + len(self._test_indices)

    def pool_is_empty(self, target_set=None):
        """
        Check if the specified pool is empty.

        :param target_set: Which set to check.
        :return: True if the pool is empty, False otherwise.
        """
        if target_set is None:
            target_set = Set.TRAIN

        return len(self._train_pool if target_set == Set.TRAIN
                   else self._test_pool) == 0
