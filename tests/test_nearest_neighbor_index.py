"""nn_search_test"""

import random
import time
import unittest
import random

from pynn import NearestNeighborIndex


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

    def test_backtracking(self):
        """
        Tests whether the search explores the opposite subtree
        when it may contain a closer point.
        """
        points = [
            (0, 0),
            (5, 5),
            (6, 0),
        ]

        uut = NearestNeighborIndex(points)

        self.assertEqual((6, 0), uut.find_nearest((4, 0)))

    def test_duplicate_points(self):
        """
        Tests whether duplicate points are indexed correctly.
        """
        points = [
            (1, 2),
            (1, 2),
            (1, 2),
            (10, 10),
        ]

        uut = NearestNeighborIndex(points)

        self.assertEqual((1, 2), uut.find_nearest((0, 0)))

    def test_identical_axis_values(self):
        """
        Tests KD-tree construction and searching when
        all points share the same x-coordinate.
        """
        points = [
            (5, 1),
            (5, 3),
            (5, 5),
            (5, 7),
            (5, 9),
        ]

        uut = NearestNeighborIndex(points)

        self.assertIn(uut.find_nearest((6, 8)), [(5, 7), (5, 9)])

    def test_equidistant_points(self):
        """
        Tests whether either equally close point can
        be returned as a valid nearest neighbor.
        """
        points = [
            (-1, 0),
            (1, 0),
        ]

        uut = NearestNeighborIndex(points)

        result = uut.find_nearest((0, 0))

        self.assertIn(result, points)

    def test_single_point(self):
        """
        Tests an index containing only one point.
        """
        uut = NearestNeighborIndex([(5, 10, 15)])

        self.assertEqual(
            (5, 10, 15),
            uut.find_nearest((100, 200, 300)),
        )

    def test_randomized_ndim(self):
        """
        Compares KD-tree results against brute-force searches
        across datasets with varying dimensions.
        """
        rng = random.Random(42)
        for dim in (1, 2, 3, 4, 8, 16):
            points = [
                tuple(rng.uniform(-1000, 1000) for _ in range(dim)) for _ in range(100)
            ]

            uut = NearestNeighborIndex(points)

            for _ in range(100):
                query = tuple(rng.uniform(-1000, 1000) for _ in range(dim))

                expected = uut.find_nearest_slow_ndim(
                    query_point=query, haystack=points
                )

                actual = uut.find_nearest(query)

                self.assertEqual(
                    expected,
                    actual,
                    f"Failed for dimension {dim}, query {query}",
                )
