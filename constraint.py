from keras import backend as K

class Constraint(object):

    def __call__(self, w):
        return w

    def get_config(self):
        return {}

class MyConstraint(Constraint):
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, w):
        '''
        #softmax
        e = K.exp(w - K.max(w,  axis=self.axis, keepdims=True))
        s = K.sum(e,  axis=self.axis, keepdims=True)
        return e / s
        '''

        w *= K.cast(K.greater_equal(w, 0.), K.floatx())  # non negative
        return w / K.sum(w, axis=self.axis, keepdims=True)

    def get_config(self):
        return {'axis': self.axis}