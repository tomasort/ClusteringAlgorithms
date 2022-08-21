class Cluster:
    def __init__(self, center=None, points=None):
        if points is None:
            points = []
        self.center = center
        self.points = points

    def add(self, data_point, update_center=True):
        self.points.append(data_point)
        if update_center:
            # compute the new center
            self.update_center()

    def update_center(self):
        new_center = []
        if len(self.points) > 0:
            for coord in range(len(self.center)):
                center_coord = sum([x[coord]
                                   for x in self.points])/len(self.points)
                new_center.append(center_coord)
            self.center = new_center

    def __repr__(self):
        return "Cluster: " + "points=" + self.points.__repr__() + " center=" + self.center.__repr__()

    def __eq__(self, other):
        if other is None:
            return False
        return self.center == other.center
