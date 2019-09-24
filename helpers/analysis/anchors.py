import numpy as np


def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return np.array(similarities)


def avg_IOU(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum += max(IOU(X[i], centroids))
    return sum / n


def write_anchors_to_file(centroids, X, anchor_file):
    f = open(anchor_file, 'w')

    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0] *= images_width / 32.
        anchors[i][1] *= images_height / 32.

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

    # there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1]))

    f.write('%f\n' % (avg_IOU(X, centroids)))
    print()


def kmeans(X, centroids, eps, anchor_file):
    N = X.shape[0]
    iterations = 0
    k, dim = centroids.shape
    prev_assignments = np.ones(N) * (-1)
    iter = 0
    old_D = np.zeros((N, k))

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))

        # assign samples to centroids
        assignments = np.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            print("Centroids = ", centroids)
            write_anchors_to_file(centroids, X, anchor_file)
            return

        # calculate new centroids
        centroid_sums = np.zeros((k, dim), np.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j))

        prev_assignments = assignments.copy()
        old_D = D.copy()
