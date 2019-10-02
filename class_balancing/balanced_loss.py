from tensorflow import keras

# https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
def weighted_sparse_categorical_crossentropy(weights, n_classes):
    weights = keras.backend.variable(weights)
        
    def loss(y_true, y_pred):
        # Convert the input from sparse classes to one-hot encoding
        # TODO: Optimize!
        y_true = keras.backend.cast(y_true, dtype='int32')
        y_true = keras.backend.one_hot(y_true, n_classes)

        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= keras.backend.sum(y_pred, axis=-1, keepdims=True)

        # clip to prevent NaN's and Inf's
        y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())

        # calc
        loss = y_true * keras.backend.log(y_pred) * weights
        loss = -keras.backend.sum(loss, -1)
        return loss
    
    return loss
