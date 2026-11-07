import math


class NearestNeighborIndex:
    """
    TODO give me a decent comment

    NearestNeighborIndex is intended to index a set of provided points to provide fast nearest
    neighbor lookup. For now, it is simply a stub that performs an inefficient traversal of all
    points every time.
    """

    def __init__(self, points):
        """
        takes an array of 2d tuples as input points and sorts array by x-coordinates.
        """
        self.points = sorted(points, key=lambda p: p[0])

    @staticmethod
    def find_nearest_slow(query_point, haystack):
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

    def bisect_left_by_x_coord(self, target):
        """
        bisect_left returns the leftmost insertion point for target based on the x-coordinate of each point.
        """
        left = 0
        right = len(self.points) - 1
        while right - left > 1:
            middle = (right + left) // 2
            if self.points[middle][0] >= target:
                right = middle
            else:
                left = middle
        return right

    def find_nearest_fast(self, query_point):
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
                point = self.points[left]
                dist_x = point[0] - query_x
                dist_x_squared = dist_x * dist_x
                if dist_x_squared > min_dist_squared:
                    left = -1
                else:
                    dist_y = point[1] - query_y
                    dist_squared = dist_x_squared + dist_y * dist_y
                    if dist_squared < min_dist_squared:
                        min_dist_squared = dist_squared
                        min_point = point
                    left -= 1
            if right < len(self.points):
                point = self.points[right]
                dist_x = point[0] - query_x
                dist_x_squared = dist_x * dist_x
                if dist_x_squared > min_dist_squared:
                    right = len(self.points)
                else:
                    dist_y = point[1] - query_y
                    dist_squared = dist_x_squared + dist_y * dist_y
                    if dist_squared < min_dist_squared:
                        min_dist_squared = dist_squared
                        min_point = point
                    right += 1

        return min_point

    def find_nearest(self, query_point):
        """
        TODO comment me.
        """

        # TODO implement me so this class runs much faster.
        return self.find_nearest_fast(query_point)
