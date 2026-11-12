import csv
import math


class Node:
    def __init__(self, point=None, left=None, right=None, dim=None):
        self.point = point
        self.left = left
        self.right = right
        self.dim = dim


class NearestNeighborIndex:
    """
    NearestNeighborIndex indexes a set of given points and provides fast nearest
    neighbor lookup through find_nearest.
    Points are stored in a KD-tree which allows for efficient search in a K-dimensional space.
    https://www.geeksforgeeks.org/dsa/search-and-insertion-in-k-dimensional-tree/
    """

    def __init__(self, points=None, path=""):
        """
        Initializes a KD-tree for efficient nearest-neighbor searches
        in K-dimensional space.

        Points can be provided as a list of coordinate tuples, a path to
        a CSV file, or both. When both are provided, the points loaded from
        the CSV file are appended to the supplied list.

        The CSV file must contain a header row followed by rows of numeric
        coordinates. Each row represents a single point, and each column
        represents a dimension. The first row of files are reserved for
        headers and are skipped.

        All points must have the same number of dimensions. After loading,
        the points are validated and organized into a KD-tree.

        Args:
        points (list[Point]): A list of K-dimensional points, where each
            point is a tuple of integers or floats. Defaults to an empty list.
        path (str): Path to a CSV file containing points. Defaults to an
            empty string, indicating that no CSV file will be loaded.
        """
        self.root = None
        self.points = []
        if isinstance(points, list):
            self.points = points
        self.dim_count = 0

        if path != "":
            # appends all points in found in file
            with open(path, "r", newline="", encoding="utf-8") as file:
                reader = csv.reader(file)
                i = 0
                for row in reader:
                    if i == 0:
                        i += 1
                        continue
                    mapped_row = tuple(map(float, row))
                    self.points.append(mapped_row)

        if len(self.points) > 0:
            self.dim_count = len(self.points[0])

            # ensures all points have the same number of dimensions
            for point in self.points:
                if len(point) != self.dim_count:
                    raise ValueError("All points must have the same dimensions")

            # builds the KD-tree
            self.root = self.build_kd_tree(self.points, 0)

    def build_kd_tree(self, points, depth) -> Node | None:
        """
        Recursively constructs a balanced KD-tree by partitioning points along
        alternating dimensions using the median at each depth.

        Args:
            points: List of points to organize into the KD-tree.
            depth: Current depth of the tree, used to determine the splitting dimension.

        Returns:
            The root node of the constructed subtree, or None if no points are provided.
        """
        if len(points) == 0:
            return None

        # sorts points by splitting dimension
        dim = depth % self.dim_count
        sorted_points = sorted(points, key=lambda p: p[dim])

        # selects the median point along the current splitting dimension
        median_idx = len(sorted_points) // 2
        median = sorted_points[median_idx]

        node = Node(median)
        node.dim = dim

        # recursively builds the left and right subtrees from points partitioned by the median
        node.left = self.build_kd_tree(sorted_points[:median_idx], depth + 1)
        node.right = self.build_kd_tree(sorted_points[median_idx + 1 :], depth + 1)

        return node

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

    @staticmethod
    def find_nearest_slow_ndim(query_point, haystack):
        """
        Finds the nearest point using a brute-force search
        over K-dimensional points.
        """

        min_dist = None
        min_point = None

        for point in haystack:
            dist = NearestNeighborIndex.find_distance(query_point, point)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_point = point

        return min_point

    @staticmethod
    def find_distance_squared(p1, p2):
        """
        Calculates the squared Euclidean distance between two K-dimensional points.

        Args:
            p1: First point.
            p2: Second point.

        Returns:
            The sum of squared differences between corresponding coordinates.

        Raises:
            ValueError: If the points have different dimensions.
        """
        if len(p1) != len(p2):
            raise ValueError("Points must have the same dimensions")

        dist_squared = 0
        for i in range(len(p1)):
            dist_i = p1[i] - p2[i]
            dist_squared += dist_i * dist_i

        return dist_squared

    @staticmethod
    def find_distance(p1, p2):
        """
        Calculates the Euclidean distance between two K-dimensional points.

        Args:
            p1: First point.
            p2: Second point.

        Returns:
            The sum of squared differences between corresponding coordinates.

        Raises:
            ValueError: If the points have different dimensions.
        """
        return math.sqrt(NearestNeighborIndex.find_distance_squared(p1, p2))

    def find_nearest_fast(self, query_point, node):
        """
        Recursively finds the nearest neighbor to a query point using KD-tree search.

        Searches the subtree containing the query point first, then explores the
        opposite subtree only if it could contain a closer point.

        Args:
            query_point: The K-dimensional point for which to find the nearest neighbor.
            node: Root node of the subtree to search.

        Returns:
            The node containing the nearest point, or None if the subtree is empty.
        """
        if node is None:
            return None

        # determines which subtree to search first based on the splitting dimension
        if query_point[node.dim] < node.point[node.dim]:
            next_branch = node.left
            other_branch = node.right
        else:
            next_branch = node.right
            other_branch = node.left

        # initializes the closest node to the current node
        closest_node = node
        closest_node_dist_sq = self.find_distance_squared(node.point, query_point)

        # searches for the closest node in the next_branch subtree
        temp_node = self.find_nearest_fast(query_point, next_branch)

        # updates the closest node if a closer node is found
        if temp_node is not None:
            temp_node_dist_sq = self.find_distance_squared(temp_node.point, query_point)
            if temp_node_dist_sq < closest_node_dist_sq:
                closest_node = temp_node
                closest_node_dist_sq = temp_node_dist_sq

        # calculcates the distance to the splitting hyperplane
        dim_dist = query_point[node.dim] - node.point[node.dim]

        # if the squared distance to the splitting hyperplane is less the squared distance to the current node
        # then the other side of the hyperplane is worth checking for a closer node
        if closest_node_dist_sq > dim_dist * dim_dist:
            # searches for the closest point in the other_branch subtree
            temp_node = self.find_nearest_fast(query_point, other_branch)

            # updates the closest node if a closer node is found
            if temp_node is not None:
                temp_node_dist_sq = self.find_distance_squared(
                    temp_node.point, query_point
                )
                if temp_node_dist_sq < closest_node_dist_sq:
                    closest_node = temp_node
                    closest_node_dist_sq = temp_node_dist_sq

        return closest_node

    def find_nearest(self, query_point):
        """
        Finds the nearest indexed point to the query point using KD-tree search
        and Euclidean distance.

        Args:
            query_point: The K-dimensional point for which to find the nearest neighbor.

        Returns:
            The closest indexed point, or None if the KD-tree is empty.
        """
        node = self.find_nearest_fast(query_point, self.root)
        if node is not None:
            return node.point
        return None


if __name__ == "__main__":
    pass
