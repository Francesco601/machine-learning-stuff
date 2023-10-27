import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, SGD

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(28*28,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_test, y_test))

model_sgd = Sequential()
model_sgd.add(Dense(512, activation='relu', input_shape=(28*28,)))
model_sgd.add(Dense(512, activation='relu'))
model_sgd.add(Dense(10, activation='softmax'))

model_sgd.compile(loss='categorical_crossentropy',
                  optimizer=SGD(),
                  metrics=['accuracy'])

history_sgd = model_sgd.fit(x_train, y_train,
                            batch_size=128,
                            epochs=10,
                            verbose=1,
                            validation_data=(x_test, y_test))


print("RMSprop:")
print("Test loss:", history.history['val_loss'][-1])
print("Test accuracy:", history.history['val_accuracy'][-1])

print("\nVanilla Gradient Descent:")
print("Test loss:", history_sgd.history['val_loss'][-1])
print("Test accuracy:", history_sgd.history['val_accuracy'][-1])
