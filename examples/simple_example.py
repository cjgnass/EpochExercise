# Please create a simple example use of the pynn library for your end user. Assume that the end
# user knows a lot about their subject matter but has only a basic understanding of Python.

# Meaningful examples may include reading a file, finding a few nearby points and writing them
# out to the console.

from pynn.nearest_neighbor_index import NearestNeighborIndex


def simple_example():
    points = [
        (-10, 10),
        (-10, 10),
        (10, -10),
        (10, 10),
    ]
    uut = NearestNeighborIndex(points=points)
    query_point = (1, 1)
    print(uut.find_nearest(query_point))


def read_from_file_example():
    uut = NearestNeighborIndex(path="points_1000.csv")
    query_point = (100, 1)
    print(uut.find_nearest(query_point))


if __name__ == "__main__":
    # simple_example()
    read_from_file_example()
