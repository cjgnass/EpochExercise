# Please create a simple example use of the pynn library for your end user. Assume that the end
# user knows a lot about their subject matter but has only a basic understanding of Python.

# Meaningful examples may include reading a file, finding a few nearby points and writing them
# out to the console.

from pynn.nearest_neighbor_index import NearestNeighborIndex


def examples():
    """
    Demonstrates how to use NearestNeighborIndex to find the closest point
    to a given location in a K-dimensional space.

    NearestNeighborIndex allows you to efficiently search a collection of
    points to identify the one closest to a specified query point using
    Euclidean distance (straight-line distance).

    Steps:
        1. Define a list of points, where each point contains coordinates
           representing its position in K-dimensional space
        2. Create a NearestNeighborIndex instance using a list of points,
           a path to a CSV file containing points, or both.
        3. Define a query point representing the location of interest.
        4. Call find_nearest() to retrieve the indexed point closest to
           the query point.
        5. Print the result.

    Notes:
        - All points must have the same number of dimensions
        - First line of files are reserved for headers and are skipped
    """

    # Example 1.
    # Defines a list of points in 2D space
    points = [
        (-10, -10),
        (-10, 10),
        (10, -10),
        (10, 10),
    ]
    # Creates a NearestNeighborIndex instance using the list of points
    uut = NearestNeighborIndex(points=points)
    # Defines query point in 2D space
    query_point = (5, 5)
    # Retrieves the indexed point closest to the query point
    closest_point = uut.find_nearest(query_point)
    # prints the result
    print(f"Example 1. Closest Point: {closest_point}, Query Point: {query_point}")

    # Example 2.
    # Defines a list of points in 3D space
    points = [
        (-10, -10, -10),
        (-10, 10, 0),
        (10, -10, 10),
        (10, 10, 100),
    ]
    # Creates a NearestNeighborIndex instance using the list of points
    uut = NearestNeighborIndex(points=points)
    # Defines query point in 3D space
    query_point = (5, 5, 0)
    # Retrieves the indexed point closest to the query point
    closest_point = uut.find_nearest(query_point)
    # prints the result
    print(f"Example 2. Closest Point: {closest_point}, Query Point: {query_point}")

    # Example 3.
    # Creates a NearestNeighborIndex instance using the path to csv containing points
    uut = NearestNeighborIndex(path="points_1000.csv")
    # Defines query point in 2D space
    query_point = (5, 5)
    # Retrieves the indexed point closest to the query point
    closest_point = uut.find_nearest(query_point)
    # prints the result
    print(f"Example 3. Closest Point: {closest_point}, Query Point: {query_point}")


if __name__ == "__main__":
    examples()
