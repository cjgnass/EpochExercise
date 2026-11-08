import math

type Number = int | float


class NearestNeighborIndex:
    """
    NearestNeighborIndex indexes a set of given points and provides fast nearest
    neighbor lookup through find_nearest_fast.
    """

    def __init__(self, points: list[tuple[Number, Number]]):
        """
        takes an array of 2d tuples as input points and sorts array by x-coordinates.
        """
        self.points: list[tuple[Number, Number]] = sorted(points, key=lambda p: p[0])

    @staticmethod
    def find_nearest_slow(
        query_point: tuple[Number, Number], haystack: list[tuple[Number, Number]]
    ) -> tuple[Number, Number] | None:
        """
        find_nearest_slow returns the point that is closest to query_point. If there are no indexed
        points, None is returned.
        """

        min_dist = None
        min_point = None

        for point in haystack:
            deltax = point[0] - query_point[0]
            deltay = point[1] - query_point[1]
            dist = math.sqrt(deltax * deltax + deltay * deltay)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_point = point

        return min_point

    def bisect_left_by_x_coord(self, target: Number) -> int:
        """
        bisect_left returns the index of the leftmost insertion point for target based on the x-coordinate of each point.
        """
        if len(self.points) == 0:
            return 0

        left = 0
        right = len(self.points) - 1
        while right - left > 1:
            middle = (right + left) // 2
            if self.points[middle][0] >= target:
                right = middle
            else:
                left = middle

        """
        If we find the the target at the left index or should be placed before the left index, 
        return left. Otherwise return right.
        """
        return left if left >= 0 and self.points[left][0] >= target else right

    def find_nearest_fast(
        self, query_point: tuple[Number, Number]
    ) -> tuple[Number, Number] | None:
        """
        TODO: Re-implement me with your faster solution.

        find_nearest_fast returns the point that is closest to query_point. If there are no indexed
        points, None is returned.
        """
        if not self.points:
            return None

        min_point = None
        min_dist_squared = float("inf")
        query_x, query_y = query_point[0], query_point[1]
        right = self.bisect_left_by_x_coord(query_x)
        left = right - 1

        while left >= 0 or right < len(self.points):
            if left >= 0:
                # checks next point to the left with the minimum x-distance to the query point
                point = self.points[left]
                dist_x = point[0] - query_x
                dist_x_squared = dist_x * dist_x

                if dist_x_squared >= min_dist_squared:
                    """
                    prunes the remaining points in the left side of self.points
                    if dist_x_squared is greater than or equal to min_dist_squared
                    """
                    left = -1

                else:
                    # calculates distance squared
                    dist_y = point[1] - query_y
                    dist_squared = dist_x_squared + dist_y * dist_y

                    if dist_squared < min_dist_squared:
                        """
                        sets min_point to the new closest point and
                        sets min_dist_squared to its corresponding distance
                        """
                        min_dist_squared = dist_squared
                        min_point = point
                    left -= 1

            if right < len(self.points):
                # checks next point to the right with the minimum x-distance to the query point
                point = self.points[right]
                dist_x = point[0] - query_x
                dist_x_squared = dist_x * dist_x

                if dist_x_squared > min_dist_squared:
                    """
                    prunes the remaining points in the right side of self.points
                    if dist_x_squared is greater than or equal to min_dist_squared
                    """
                    right = len(self.points)

                else:
                    # calculates distance squared
                    dist_y = point[1] - query_y
                    dist_squared = dist_x_squared + dist_y * dist_y
                    if dist_squared < min_dist_squared:
                        """
                        sets min_point to the new closest point and
                        sets min_dist_squared to its corresponding distance
                        """
                        min_dist_squared = dist_squared
                        min_point = point
                    right += 1

        return min_point

    def find_nearest(
        self, query_point: tuple[Number, Number]
    ) -> tuple[Number, Number] | None:
        """
        Returns the indexed point closes to the query_point using Euclidean distance.
        If no points are indexed, returns None.
        """

        # TODO implement me so this class runs much faster.
        return self.find_nearest_fast(query_point)
