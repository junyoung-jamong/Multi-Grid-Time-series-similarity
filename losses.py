import keras.backend as K
from keras import layers

class DoubletLossLayer(layers.Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(DoubletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        positive_dist, negative_dist = inputs
        max = K.maximum(positive_dist, negative_dist)
        return K.sum(K.maximum(positive_dist/max - negative_dist/max + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss