from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, TimeDistributed, LSTM


def build_video_model(input_shape=(None, 224, 224, 3), num_classes=2):
    """
    Build a CNN-LSTM model for video classification

    Parameters:
    input_shape (tuple): The shape of the input video frames (time steps, width, height, channels)
    num_classes (int): The number of classes for the output layer

    Returns:
    model: A Keras model instance
    """

    model = Sequential()

    # TimeDistributed layer to apply the same convolutional operations to each frame in the sequence
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    model.add(TimeDistributed(Flatten()))

    # LSTM layer to learn temporal dependencies between frames
    model.add(LSTM(64))

    # Dense layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid'))

    # Compile the model
    model.compile(loss='categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    model = build_video_model()
    model.summary()
