1. 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(*load_iris(return_X_y=True), test_size=0.2)
print(f"Accuracy: {RandomForestClassifier(100).fit(X_train, y_train).score(X_test, y_test):.2f}")


 2.
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
X_train, X_test = pad_sequences(X_train, maxlen=200), pad_sequences(X_test, maxlen=200)

tfidf = TfidfTransformer().fit(X_train)
X_train, X_test = tfidf.transform(X_train), tfidf.transform(X_test)

knn = KNeighborsClassifier(5).fit(X_train, y_train)
print(f'Accuracy: {accuracy_score(y_test, knn.predict(X_test))}')


3.
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

x_train = np.array([np.bincount(seq, minlength=10000) for seq in x_train])
x_test = np.array([np.bincount(seq, minlength=10000) for seq in x_test])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(46, activation='softmax')
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train[1000:], to_categorical(y_train[1000:]), epochs=25, batch_size=512, 
          validation_data=(x_train[:1000], to_categorical(y_train[:1000])))

print(f'Accuracy: {model.evaluate(x_test, to_categorical(y_test), verbose=0)[1]}')

4. 
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train.reshape(-1, 28, 28, 1)/255.0, X_test.reshape(-1, 28, 28, 1)/255.0
y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
print(f'Accuracy: {model.evaluate(X_test, y_test)[1]}')

5. 
import tensorflow as tf
import tensorflow_datasets as tfds

train_dataset, test_dataset = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:]'], as_supervised=True)
train_dataset = train_dataset.map(lambda x, y: (tf.image.resize(x, (150, 150)) / 255.0, y)).batch(32).shuffle(1000)
test_dataset = test_dataset.map(lambda x, y: (tf.image.resize(x, (150, 150)) / 255.0, y)).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=test_dataset, epochs=10)
model.evaluate(test_dataset)


6.  
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
train_padded = pad_sequences(train_data, maxlen=100)
test_padded = pad_sequences(test_data, maxlen=100)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 32, input_length=100),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels))
