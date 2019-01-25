from files import *
from utils import *
from grid import Grid
from similarity import *
from constraint import MyConstraint
import sys
from random import random
from losses import DoubletLossLayer
import keras
from keras import layers, Model
from keras.layers import Input, Lambda

def build_model(input_shape, alpha=0.2):
    positive_input_layer = Input(shape=input_shape)
    negative_input_layer = Input(shape=input_shape)

    #base_network = layers.Dense(1, input_shape=input_shape, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.01), kernel_constraint=keras.constraints.NonNeg())
    base_network = layers.Dense(1, input_shape=input_shape, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001), kernel_constraint=MyConstraint())

    positive_dist = base_network(positive_input_layer)
    negative_dist = base_network(negative_input_layer)

    doublet_loss_layer = DoubletLossLayer(alpha=alpha, name='doublet_loss_layer')([positive_dist, negative_dist])
    model = Model([positive_input_layer, negative_input_layer], doublet_loss_layer)
    #model.compile(loss=None, optimizer=keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0))
    model.compile(loss=None, optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))

    dist_input = Input(shape=input_shape)
    dist_layer = base_network(dist_input)
    dist_model = Model([dist_input], dist_layer)

    return model, dist_model

def doublet_generator(dist_table, y_trains, dist_model, threshold=0.5):
    p_batch = []
    n_batch = []

    for q_idx in range(len(dist_table)):
        query_label = y_trains[q_idx]

        base_dist_list = dist_table[q_idx]

        dist = dist_model.predict(base_dist_list).reshape(-1)

        min_dist = sys.float_info.max
        predict_label = -1
        temp = [[-1, sys.float_info.max], [-1, sys.float_info.max]]  # [[positive index, positive min distance], [negative index, negative min distance]]
        for d_idx in range(len(dist)):
            if q_idx == d_idx:
                continue
            if dist[d_idx] < min_dist:
                min_dist = dist[d_idx]
                predict_label = y_trains[d_idx]

            # if same class data
            if query_label == y_trains[d_idx]:
                if dist[d_idx] < temp[0][1]:
                    temp[0][1] = dist[d_idx]
                    temp[0][0] = d_idx
            else:
                if dist[d_idx] < temp[1][1]:
                    temp[1][1] = dist[d_idx]
                    temp[1][0] = d_idx

        if query_label != predict_label or threshold < random():
            p_batch.append(base_dist_list[temp[0][0]])
            n_batch.append(base_dist_list[temp[1][0]])

    return [p_batch, n_batch], None

if __name__ == '__main__':
    dataset = 'FaceAll'
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

    # generate distance and observe max_distance
    max_distance = sys.float_info.min
    min_distance = sys.float_info.max

    dist_table = []
    for q_idx in range(len(y_trains)):
        query_wise_sim = []
        for b_idx in range(len(y_trains)):
            grid_wise_sim = []
            for grid_idx in range(M):
                dist = GMED(grids_x_train_matrices[grid_idx][b_idx], grids_x_train_matrices[grid_idx][q_idx])
                grid_wise_sim.append(dist)
                if max_distance < dist:
                    max_distance = dist
                if min_distance > dist:
                    min_distance = dist
            query_wise_sim.append(grid_wise_sim)

        dist_table.append(query_wise_sim)
    dist_table = np.array(dist_table)

    # generate model
    model, dist_model = build_model(input_shape=(M, ), alpha=max_distance*0.05)
    #model.summary()
    #dist_model.summary()

    # model weight initialize
    model.layers[-2].set_weights(np.expand_dims(np.ones((M, 1)) / M, axis=0))
    #print(model.layers[-2].get_weights())

    for i in range(250):
        doublet = doublet_generator(dist_table, y_trains, dist_model, threshold=0.9)
        print(dist_model.layers[-1].get_weights())

        if len(doublet[0][0]) > 0:
            model.fit(doublet[0], doublet[1], epochs=1, batch_size=max(len(doublet)/5, 1))
        else:
            break

    # 1-NN classification without weights and with weights
    weights = model.layers[-2].get_weights()
    print('weights :', weights)

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


