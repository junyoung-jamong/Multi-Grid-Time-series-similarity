import keras.backend as K
from keras import layers

class DoubletLossLayer(layers.Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(DoubletLossLayer, self).__init__(**kwargs)

    def doublet_loss(self, inputs):
        positive_dist, negative_dist = inputs
        return K.sum(K.maximum(positive_dist - negative_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.doublet_loss(inputs)
        self.add_loss(loss)
        return loss