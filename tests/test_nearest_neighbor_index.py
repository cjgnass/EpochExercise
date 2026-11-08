"""nn_search_test"""

import random
import time
import unittest

from pynn import NearestNeighborIndex, Number


class NearestNeighborIndexTest(unittest.TestCase):
    def test_basic(self):
        """
        test_basic tests a handful of nearest neighbor queries to make sure they return the right
        result.
        """

        test_points = [
            (1, 2),
            (1, 0),
            (10, 5),
            (-1000, 20),
            (3.14159, 42),
            (42, 3.14159),
        ]

        uut = NearestNeighborIndex(test_points)

        self.assertEqual((1, 0), uut.find_nearest((0, 0)))
        self.assertEqual((-1000, 20), uut.find_nearest((-2000, 0)))
        self.assertEqual((42, 3.14159), uut.find_nearest((40, 3)))

    def test_benchmark(self):
        """
        test_benchmark tests a bunch of values using the slow and fast version of the index
        to determine the effective speedup.
        """

        def rand_point():
            return (random.uniform(-1000, 1000), random.uniform(-1000, 1000))

        index_points = [rand_point() for _ in range(10000)]
        query_points = [rand_point() for _ in range(1000)]

        expected = []
        actual = []

        # Run the baseline slow tests to get the expected values.
        start = time.time()
        for query_point in query_points:
            expected.append(
                NearestNeighborIndex.find_nearest_slow(query_point, index_points)
            )
        slow_time = time.time() - start

        # don't include the indexing time when benchmarking
        uut = NearestNeighborIndex(index_points)

        # Run the indexed tests
        start = time.time()
        for query_point in query_points:
            actual.append(uut.find_nearest(query_point))
        new_time = time.time() - start

        self.assertEqual(expected, actual)

        print(f"slow time: {slow_time:0.2f}sec")
        print(f"new time: {new_time:0.2f}sec")
        print(f"speedup: {(slow_time / new_time):0.2f}x")

    # TODO: Add more test cases to ensure your index works in different scenarios
    def test_no_points(self):
        """
        test_no_points tests nearest neighbor queries returns None if NearestNeighborIndex is initialized with no points.
        """

        test_points = []

        uut = NearestNeighborIndex(test_points)

        self.assertEqual(None, uut.find_nearest((0, 0)))
        self.assertEqual(None, uut.find_nearest((-2000, 0)))
        self.assertEqual(None, uut.find_nearest((40, 3)))

    def test_bisect_left_by_x_coord(self):
        """
        test_bisect_left_by_x_coord tests bisect_left_by_x_coord
        """
        # tests no points
        test_points: list[tuple[Number, Number]] = []
        uut = NearestNeighborIndex(test_points)
        self.assertEqual(0, uut.bisect_left_by_x_coord(0))

        # tests one point
        test_points: list[tuple[Number, Number]] = [(0, 0)]
        uut = NearestNeighborIndex(test_points)
        self.assertEqual(0, uut.bisect_left_by_x_coord(0))

        # tests left edge
        test_points: list[tuple[Number, Number]] = [(i, i) for i in range(10)]
        uut = NearestNeighborIndex(test_points)
        self.assertEqual(0, uut.bisect_left_by_x_coord(0))
        self.assertEqual(0, uut.bisect_left_by_x_coord(-1))

        # tests right edge
        test_points: list[tuple[Number, Number]] = [(i, i) for i in range(10)]
        uut = NearestNeighborIndex(test_points)
        self.assertEqual(9, uut.bisect_left_by_x_coord(9))
        self.assertEqual(9, uut.bisect_left_by_x_coord(10))

        # tests if multiple targets are found in list
        test_points: list[tuple[Number, Number]] = [(1, 1), (1, 1), (2, 2)]
        uut = NearestNeighborIndex(test_points)
        self.assertEqual(0, uut.bisect_left_by_x_coord(1))

        test_points: list[tuple[Number, Number]] = [(1, 1), (2, 1), (2, 2)]
        uut = NearestNeighborIndex(test_points)
        self.assertEqual(1, uut.bisect_left_by_x_coord(2))
