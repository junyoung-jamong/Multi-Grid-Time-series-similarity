from files import *
from utils import *
from grid import Grid
from similarity import *
import sys

if __name__ == '__main__':
    dataset = '50words'
    x_trains, y_trains, x_tests, y_tests = get_ucr_train_test_datasets(dataset)
    x_trains = feature_scaling_datasets(x_trains)
    x_tests = feature_scaling_datasets(x_tests)

    M = 10  # the number of Grids

    grids = []
    grids_x_train_matrices = []
    grids_x_test_matrices = []
    for i in range(M):
        g = Grid()
        grids.append(g)
        x_trains_matrices = g.dataset2Matrices(x_trains)
        grids_x_train_matrices.append(x_trains_matrices)
        x_test_matrices = g.dataset2Matrices(x_tests)
        grids_x_test_matrices.append(x_test_matrices)

    # 1-NN classification with multi-grid
    predict_cnt = 0
    error_cnt = 0
    for q_idx in range(len(y_tests)):
        query_label = y_tests[q_idx]

        min_dist = sys.float_info.max
        predict_label = -1
        for b_idx in range(len(y_trains)):
            base_label = y_trains[b_idx]

            dist = 0
            for grid_idx in range(M):
                dist += GMED(grids_x_train_matrices[grid_idx][b_idx], grids_x_test_matrices[grid_idx][q_idx])

            if dist < min_dist:
                min_dist = dist
                predict_label = base_label

        predict_cnt += 1
        if query_label != predict_label:
            error_cnt += 1

    print('erorr rate :', error_cnt/predict_cnt)