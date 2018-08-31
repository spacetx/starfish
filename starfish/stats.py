import regional
from scipy.sparse import coo_matrix


def label_to_regions(labels) -> regional.many:
    label_mat_coo = coo_matrix(labels)

    def region_for(label_mat_coo, label):
        ind = label_mat_coo.data == label
        # TODO does this work in 3D?
        x = label_mat_coo.row[ind]
        y = label_mat_coo.col[ind]

        re = regional.one(list(zip(x, y)))
        return re

    unique_labels = sorted(set(label_mat_coo.data))
    regions = [region_for(label_mat_coo, label) for label in unique_labels]

    return regional.many(regions)
