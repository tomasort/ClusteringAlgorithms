import argparse
import pandas as pd
import sys
from kmeans import KMeans
from knn import KNN


def read_file(filename):
    # finally reading the file is the easiest part of the assignment and not the most laborious
    return pd.read_csv(filename, header=None)


def e2(p1, p2):
    distance = 0
    for p1_, p2_ in zip(p1, p2):
        distance += (p1_ - p2_) ** 2
    # print(f"The distance between {p1} and {p2} is {distance}")
    return distance


def manh(p1, p2):
    distance = 0
    for p1_, p2_ in zip(p1, p2):
        distance += abs(p1_ - p2_)
    return distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Implementation of KNN and K-means')
    parser.add_argument('-mode', nargs='?', choices=['knn', 'kmeans'], type=str,
                        required=True, help="knn or kmeans. This flag sets the algorithm that will be used")
    parser.add_argument('-k', nargs='?', type=int, required=False, default=3,
                        help="k value to use. This means the number of clusters. It must be an integer. the default is 3.")
    parser.add_argument('-d', nargs='?', choices=['e2', 'manh'], required=False, type=str, default='e2',
                        help='Distance function to use. The possible values are: 1) e2 which represents the euclidean squared distance. And 2) manh which represents the manhattan distance.')
    parser.add_argument('-unitw', action='store_true', required=False,
                        help='If present, unit voting weights will be used, if not, then 1/d weights will be used.')
    parser.add_argument('-train', nargs='?', required=False,
                        type=str, help='a file containing training data.')
    parser.add_argument('-test', nargs='?', required=False,
                        type=str, help='a file containing points to test.')
    parser.add_argument('-data', nargs='?', required=False,
                        type=str, help='a file containing data to cluster.')
    parser.add_argument('-alt', required=False, action='store_true',
                        help='if present, the program will display a message if there is ambiguity in the input files and some labels could have been different. This is only valid in KNN mode. ADDED BECAUSE OF AMBIGUITY IN PROJECT. (YES, THERE IS STILL AMBIGUITY IF NOTHING HAS CHANGED)')
    parser.add_argument('-debug', required=False, action='store_true',
                        help='simple flag for debugging purposes. NOT PART OF THE PROJECT SO DONT USE IT IF YOU ARE NOT DEBUGGING!')
    parser.add_argument('centroids', nargs='*',
                        help='initial centroids. They will only be used if the mode is kmeans')
    args = parser.parse_args(sys.argv[1:])
    mode = args.mode
    if mode == 'knn':
        # Make sure that we have a train file and a test file
        if not args.test or not args.train:
            print("Test and Train files required for knn")
            sys.exit(-1)
        train_data = read_file(args.train)
        test_data = read_file(args.test)
        model = KNN(k=args.k, train_data=train_data.values, unitw=args.unitw, dist_func=(
            manh if args.d == 'manh' else e2), show_alternative=args.alt)
        model.test(test_data.values, print_pr=True, verbose=True)
    elif mode == 'kmeans':
        if not args.data:
            print("Data file required for kmeans")
            sys.exit(-1)
        data = read_file(args.data)
        # Look for the centroids  in the input argument
        if not args.centroids:
            print("Initial centroids need to be provided in the input")
            sys.exit(-1)
        centroids = [[int(c) for c in x.replace(")", "").replace(
            "(", "").split(",")] for x in args.centroids]
        model = KMeans(centroids=centroids, dist_func=(
            manh if args.d == 'manh' else e2))
        model.train(data.values)
        for i, cluster in enumerate(model.clusters):
            lb, rb = '{', '}'
            print(
                f"C{i+1} = {lb}{','.join([str(p[-1]) for p in cluster.points])}{rb}")
        for cluster in model.clusters:
            print(f"({str(cluster.center).replace(',', '')})")
    else:
        print(f"Mode {mode} not recognized")
        sys.exit(-1)
