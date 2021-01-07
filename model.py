from tensorflow import keras


def MLTD_model(data_dim=3, seq_length=48):
    activation_function = keras.activations.tanh
    model = keras.models.Sequential()
    model.add(keras.layers.GRU(108, input_shape=(seq_length, data_dim), return_sequences=True, activation=activation_function))
    model.add(keras.layers.GRU(216, return_sequences=False, activation=activation_function))
    model.add(keras.layers.Dense(216, activation=activation_function))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(216, activation=activation_function))
    model.add(keras.layers.Dense(seq_length, activation=keras.activations.linear))
    # model.add(keras.layers.Reshape([seq_length, 1]))
    # model.add(keras.layers.Dense(27, activation=activation_function))
    # model.add(keras.layers.Dense(1, activation=keras.activations.linear))

    return model


if __name__=='__main__':
    model = MLTD_model()
    model.summary()
    keras.utils.plot_model(model)
