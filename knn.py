from collections import Counter
import sys


class KNN:
    def __init__(self, dist_func=lambda a, b: sum((a_ - b_) ** 2 for a_, b_ in zip(a, b)), k=3, train_data=None, unitw=False, show_alternative=False):
        self.dist = dist_func
        self.k = k
        self.train_data = train_data
        self.unitw = unitw
        self.show_alternative = show_alternative

    def train(self, training_data):
        self.train_data = training_data

    def test(self, testing_data, print_pr=False, verbose=False):
        if self.train_data is None:
            print("Training data must be provided")
            raise ValueError
        # go through the test data
        labels = []
        # we transform ndarray into list to list for the sort method
        data_points = list(self.train_data)
        for current_instance in testing_data:
            data_points.sort(key=lambda e: self.dist(
                current_instance[:-1], e[:-1]))
            # Look at the output and select the top k
            counter = Counter()
            if len(data_points) < self.k:
                print("There are less points than the number of clusters")
                sys.exit(1)
            if self.unitw:
                counter = Counter([closest[-1]
                                  for closest in data_points[:self.k]])
            else:
                for data_point in data_points[:self.k]:
                    if data_point[-1] not in counter.keys():
                        counter[data_point[-1]] = 0
                    # if debug:
                    #     print(
                    #         f"The distance between {data_point} and {current_instance} is {self.dist(data_point[:-1], current_instance[:-1])} and 1/d={1/max(self.dist(data_point[:-1], current_instance[:-1]), 0.0001)}")
                    counter[data_point[-1]] += 1 / \
                        max(self.dist(data_point[:-1],
                            current_instance[:-1]), 1e-4)
            max_label = counter.most_common(1)[0][0]
            ambiguity = False
            # In case there is ambiguity, we might want to print it
            if len(counter.most_common()) > 1 and len(data_points) > self.k + 1:
                top_neighbors = [closest for closest in
                                 [(self.dist(m[:-1], current_instance[:-1]), m[-1]) for m in data_points[:self.k + 1]]]
                distance_is_equal = top_neighbors[self.k - 1][0] == top_neighbors[self.k][0] and \
                    top_neighbors[self.k - 1][1] != top_neighbors[self.k][1]
                diff_is_one = counter.most_common(
                    2)[0][1]-counter.most_common(2)[1][1] == 1
                last_one_wold_change_result = top_neighbors[self.k][1] == counter.most_common(2)[
                    1][0]
                if distance_is_equal and diff_is_one and last_one_wold_change_result:
                    # if debug:
                    #     # This line will display the top k+1 neighbors with the distance to the point we are trying to label the format is: (dist, label)
                    #     print(top_neighbors)
                    ambiguity = True
            # explanation = f" OR {counter.most_common(2)[1][0]}. Both the kth and k+1th have dist={top_neighbors[self.k-1][0]}" if ambiguity else ""
            # print(
            #     f"want={current_instance[-1]} got={max_label}{explanation if self.show_alternative or debug else ''}")
            labels.append(max_label)
        if print_pr:
            for c in sorted(set(self.train_data[:, -1])):
                KNN.print_precision_and_recall(c, labels, testing_data[:, -1])
        return labels

    @staticmethod
    def get_tp_fp_tn_fn(class_, labels, truth):
        true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0
        for label, real in zip(labels, truth):
            true_positives += 1 if real == label and real == class_ else 0
            false_positives += 1 if label == class_ and real != class_ else 0
            true_negatives += 1 if label != class_ and real != class_ else 0
            false_negatives += 1 if label != class_ and real == class_ else 0
        return true_positives, false_positives, true_negatives, false_negatives

    @staticmethod
    def recall(class_, labels, truth):
        tp, fp, _, fn = KNN.get_tp_fp_tn_fn(class_, labels, truth)
        return tp/(tp + fn)

    @staticmethod
    def precision(class_, labels, truth):
        tp, fp, _, _ = KNN.get_tp_fp_tn_fn(class_, labels, truth)
        return tp/(tp + fp)

    @staticmethod
    def print_precision_and_recall(class_, labels, truth):
        tp, fp, _, fn = KNN.get_tp_fp_tn_fn(class_, labels, truth)
        print(f"Label={class_} Precision={tp}/{tp + fp} Recall={tp}/{tp + fn}")
