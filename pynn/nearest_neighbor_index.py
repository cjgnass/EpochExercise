import math

type Number = int | float


class Node:
    def __init__(self, point=None, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right


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

        self.root = self.build_kd_tree(points, 0)

    def build_kd_tree(self, points, depth) -> Node | None:
        """
        recursively builds a 2D-tree.
        """
        if len(points) == 0:
            return None
        sorted_points = sorted(points, key=lambda p: p[depth % 2])
        median_idx = len(sorted_points) // 2
        median = sorted_points[median_idx]
        node = Node(median)
        node.left = self.build_kd_tree(sorted_points[:median_idx], depth + 1)
        node.right = self.build_kd_tree(sorted_points[median_idx + 1 :], depth + 1)
        return node

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

    @staticmethod
    def find_squared_distance(p1: tuple[Number, Number], p2: tuple[Number, Number]):
        x_dist = p1[0] - p2[0]
        y_dist = p1[1] - p2[1]
        return x_dist * x_dist + y_dist * y_dist

    def find_nearest_fast(
        self, query_point: tuple[Number, Number], node: Node | None, depth: int
    ) -> Node | None:
        """
        TODO: Re-implement me with your faster solution.

        find_nearest_fast returns the point that is closest to query_point. If there are no indexed
        points, None is returned.
        """
        if node is None:
            return None

        axis = depth % 2
        if query_point[axis] < node.point[axis]:
            next_branch = node.left
            other_branch = node.right
        else:
            next_branch = node.right
            other_branch = node.left

        best = node
        best_dist = self.find_squared_distance(node.point, query_point)

        temp = self.find_nearest_fast(query_point, next_branch, depth + 1)
        if temp is not None:
            temp_dist = self.find_squared_distance(temp.point, query_point)
            if temp_dist < best_dist:
                best = temp
                best_dist = temp_dist

        dist = query_point[axis] - node.point[axis]
        if best_dist >= dist * dist:
            temp = self.find_nearest_fast(query_point, other_branch, depth + 1)
            if temp is not None:
                temp_dist = self.find_squared_distance(temp.point, query_point)
                if temp_dist < best_dist:
                    best = temp
                    best_dist = temp_dist
        return best

    def find_nearest(
        self, query_point: tuple[Number, Number]
    ) -> tuple[Number, Number] | None:
        """
        Returns the indexed point closes to the query_point using Euclidean distance.
        If no points are indexed, returns None.
        """

        # TODO implement me so this class runs much faster.
        if len(self.points) == 0:
            return None

        node = self.find_nearest_fast(query_point, self.root, 0)
        if node is not None:
            return node.point

        return None


if __name__ == "__main__":
    pass
