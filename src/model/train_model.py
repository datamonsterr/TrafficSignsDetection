import os
import numpy as np
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential
from load_data import imgs, labels

# disable using GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


data_path = 'dataset/GTSRB/Final_Training/Images'


def split_train_val_test_data(imgs, labels):

    # Chuẩn hoá dữ liệu pixels và labels
    imgs = np.array(imgs)
    labels = keras.utils.to_categorical(labels)

    # Nhào trộn dữ liệu ngẫu nhiên
    randomize = np.arange(len(imgs))
    np.random.shuffle(randomize)
    X = imgs[randomize]
    print("X=", X.shape)
    y = labels[randomize]

    # Split the dataset, 60 for training 40 for testing
    train_size = int(X.shape[0] * 0.6)
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = y[:train_size], y[train_size:]

    val_size = int(X_val.shape[0] * 0.5)
    X_val, X_test = X_val[:val_size], X_val[val_size:]
    Y_val, Y_test = Y_val[:val_size], Y_val[val_size:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


X_train, Y_train, X_val, Y_val, X_test, Y_test = split_train_val_test_data(
    imgs, labels)


def build_model(input_shape=(64, 64, 3), filter_size=(3, 3), pool_size=(2, 2), output_size=43):
    model = Sequential([
        Conv2D(16, filter_size, activation='relu',
               input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(16, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),
        Conv2D(32, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),
        Conv2D(64, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),
        Flatten(),
        Dense(2048, activation='relu'),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(output_size, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
    model.summary()
    return model


model = build_model(input_shape=(64, 64, 3), output_size=43)

# Train model
epochs = 10
batch_size = 16

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
          validation_data=(X_val, Y_val))

model.save("traffic_sign_model.h5")

# Test with new data
print(model.evaluate(X_test, Y_test))
