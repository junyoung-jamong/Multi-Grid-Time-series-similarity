from files import *
from utils import *
from grid import Grid
from similarity import *
from constraint import MyConstraint
import sys
from losses import DoubletLossLayer
import keras
from keras import layers, Model
from keras.layers import Input, Lambda

def build_model(input_shape):
    positive_input_layer = Input(shape=input_shape)
    negative_input_layer = Input(shape=input_shape)

    base_network = layers.Dense(1, input_shape=input_shape, use_bias=False, kernel_constraint=MyConstraint())

    positive_dist = base_network(positive_input_layer)
    negative_dist = base_network(negative_input_layer)

    doublet_loss_layer = DoubletLossLayer(alpha=0.1, name='doublet_loss_layer')([positive_dist, negative_dist])
    model = Model([positive_input_layer, negative_input_layer], doublet_loss_layer)
    model.compile(loss=None, optimizer='adam')

    dist_input = Input(shape=input_shape)
    dist_layer = base_network(dist_input)
    dist_model = Model([dist_input], dist_layer)

    return model, dist_model

def doublet_generator(dist_table, y_trains, dist_model):
    p_batch = []
    n_batch = []

    for q_idx in range(len(dist_table)):
        query_label = y_trains[q_idx]

        base_sim_list = dist_table[q_idx]

        dist = dist_model.predict(base_sim_list).reshape(-1)

        min_dist = sys.float_info.max
        predict_label = -1
        temp = [[-1, sys.float_info.max], [-1, sys.float_info.max]]  # [[positive index, positive min distance], [negative index, negative min distance]]
        for d_idx in range(len(dist)):
            if q_idx == d_idx:
                continue
            if dist[d_idx] < min_dist:
                min_dist = dist[d_idx]
                predict_label = y_trains[d_idx]

            if query_label == y_trains[d_idx]:
                if dist[d_idx] < temp[0][1]:
                    temp[0][1] = dist[d_idx]
                    temp[0][0] = d_idx
            else:
                if dist[d_idx] < temp[1][1]:
                    temp[1][1] = dist[d_idx]
                    temp[1][0] = d_idx

        if query_label != predict_label:
            #print('pos_dist :', temp[0][1], 'neg_dist :', temp[1][1])
            p_batch.append(base_sim_list[temp[0][0]])
            n_batch.append(base_sim_list[temp[1][0]])

    return [p_batch, n_batch], None

if __name__ == '__main__':
    dataset = 'Gun_Point'
    x_trains, y_trains, x_tests, y_tests = get_ucr_train_test_datasets(dataset)
    x_trains = feature_scaling_datasets(x_trains)
    x_tests = feature_scaling_datasets(x_tests)

    M = 5  # the number of Grids

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

    # generate distance
    dist_table = []
    for q_idx in range(len(y_trains)):
        query_wise_sim = []
        for b_idx in range(len(y_trains)):
            grid_wise_sim = []
            for grid_idx in range(M):
                dist = GMED(grids_x_train_matrices[grid_idx][b_idx], grids_x_train_matrices[grid_idx][q_idx])
                grid_wise_sim.append(dist)
            query_wise_sim.append(grid_wise_sim)

        dist_table.append(query_wise_sim)
    dist_table = np.array(dist_table)

    model, dist_model = build_model(input_shape=(M,))
    #model.summary()
    #dist_model.summary()

    model.layers[-2].set_weights(np.expand_dims(np.ones((M, 1)) / M, axis=0))
    #print(model.layers[-2].get_weights())

    model.compile(loss=None, optimizer='adam')

    for i in range(5):
        doublet = doublet_generator(dist_table, y_trains, dist_model)
        print(dist_model.layers[-1].get_weights())
        if len(doublet[0][0]) > 0:
            model.fit(doublet[0], doublet[1], epochs=500)
        else:
            break

    # 1-NN classification without weights and with weights
    weights = model.layers[-2].get_weights()

    predict_cnt = 0
    error_cnt = 0
    weighted_error_cnt = 0

    for q_idx in range(len(y_tests)):
        query_label = y_tests[q_idx]

        predict_label = -1
        weighted_predict_label = -1
        min_dist = sys.float_info.max
        min_weighted_dist = sys.float_info.max

        for b_idx in range(len(y_trains)):
            dist_sum = 0
            weighted_dist_sum = 0
            for grid_idx in range(M):
                dist = GMED(grids_x_train_matrices[grid_idx][b_idx], grids_x_test_matrices[grid_idx][q_idx])
                dist_sum += dist
                weighted_dist_sum += dist * weights[0][grid_idx]

            if dist_sum < min_dist:
                min_dist = dist_sum
                predict_label = y_trains[b_idx]
            if weighted_dist_sum < min_weighted_dist:
                min_weighted_dist = weighted_dist_sum
                weighted_predict_label = y_trains[b_idx]

        predict_cnt += 1
        if query_label != predict_label:
            error_cnt += 1
        if query_label != weighted_predict_label:
            weighted_error_cnt += 1

    print('no weight error rate :', error_cnt/predict_cnt)
    print('weighted error rate :', weighted_error_cnt/predict_cnt)


