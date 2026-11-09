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

    def test_worst_case_benchmark(self):
        """
        test_benchmark tests a bunch of values using the slow and fast version of the index
        to determine the effective speedup.
        """

        def rand_point_close_to_origin():
            return (random.uniform(-10, 10), random.uniform(-10, 10))

        def rand_far_point():
            return (
                10,
                random.choice(
                    [random.uniform(-11000, -10000), random.uniform(10000, 11000)]
                ),
            )

        index_points: list[tuple[Number, Number]] = [
            rand_far_point() for _ in range(9999)
        ]
        index_points.append((50, 50))
        query_points = [rand_point_close_to_origin() for _ in range(1000)]

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

        print()
        print("Worst Case:")
        print(f"slow time: {slow_time:0.2f}sec")
        print(f"new time: {new_time:0.2f}sec")
        print(f"speedup: {(slow_time / new_time):0.2f}x")

    def test_benchmark_ndim(self):
        """
        test_benchmark tests a bunch of values using the slow and fast version of the index
        to determine the effective speedup.
        """

        def rand_point():
            return (
                random.uniform(-1000, 1000),
                random.uniform(-1000, 1000),
                random.uniform(-1000, 1000),
                random.uniform(-1000, 1000),
            )

        index_points = [rand_point() for _ in range(10000)]
        query_points = [rand_point() for _ in range(1000)]

        expected = []
        actual = []

        # Run the baseline slow tests to get the expected values.
        start = time.time()
        for query_point in query_points:
            expected.append(
                NearestNeighborIndex.find_nearest_slow_ndim(query_point, index_points)
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

        print()
        print("N-Dim Case:")
        print(f"slow time: {slow_time:0.2f}sec")
        print(f"new time: {new_time:0.2f}sec")
        print(f"speedup: {(slow_time / new_time):0.2f}x")

    # def test_big_benchmark_ndim(self):
    #     """
    #     test_benchmark tests a bunch of values using the slow and fast version of the index
    #     to determine the effective speedup.
    #     """
    #
    #     def rand_point():
    #         return (
    #             random.uniform(-1000, 1000),
    #             random.uniform(-1000, 1000),
    #         )
    #
    #     index_points = [rand_point() for _ in range(100000)]
    #     query_points = [rand_point() for _ in range(1000)]
    #
    #     expected = []
    #     actual = []
    #
    #     # Run the baseline slow tests to get the expected values.
    #     start = time.time()
    #     for query_point in query_points:
    #         expected.append(
    #             NearestNeighborIndex.find_nearest_slow_ndim(query_point, index_points)
    #         )
    #     slow_time = time.time() - start
    #
    #     # don't include the indexing time when benchmarking
    #     uut = NearestNeighborIndex(index_points)
    #
    #     # Run the indexed tests
    #     start = time.time()
    #     for query_point in query_points:
    #         actual.append(uut.find_nearest(query_point))
    #     new_time = time.time() - start
    #
    #     self.assertEqual(expected, actual)
    #
    #     print()
    #     print("N-Dim Case:")
    #     print(f"slow time: {slow_time:0.2f}sec")
    #     print(f"new time: {new_time:0.2f}sec")
    #     print(f"speedup: {(slow_time / new_time):0.2f}x")
