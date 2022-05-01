from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from matplotlib import pyplot
import csv
import numpy as np


def prepare_data(model_type):
    print("Started data preparation...........")
    csvFile = open('responses.csv', 'r')
    features = csv.reader(csvFile, delimiter=',')
    y = []
    x = []
    for line in features:
        if (model_type == "Learning-Style"):
            x.append(line[4:14])
            y.append(line[-2])
        else:
            x.append(line[14:24])
            y.append(line[-1])

    le = LabelEncoder()
    encoded_Y = le.fit_transform(y)
    y = to_categorical(encoded_Y)

    list_x = np.array(x)
    x = list_x.astype(np.int)

    n_train = round(len(y) * 0.67)
    trainy = []
    testy = []
    trainX = []
    testX = []

    for line in y:
        if len(trainy) < n_train:
            trainy.append(line)
        else:
            testy.append(line)

    for line in x:
        if len(trainX) < n_train:
            trainX.append(line)
        else:
            testX.append(line)

    trainX = np.array(trainX)
    testX = np.array(testX)
    trainy = np.array(trainy)
    testy = np.array(testy)
    print("Finished data preparation...........")
    return trainX, trainy, testX, testy, le


def learning_model(trainX, trainy, testX, testy):
    model = Sequential()
    model.add(Dense(100, input_dim=10, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(trainX, trainy, validation_data=(testX, testy), batch_size=1, epochs=100, verbose=1)

    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.show()
    scores = model.evaluate(trainX, trainy, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    model.save("learning_model.h5")
    print("Model saved to disk")


def personality_model(trainX, trainy, testX, testy):
    model = Sequential()
    model.add(Dense(100, input_dim=10, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(trainX, trainy, validation_data=(testX, testy), batch_size=1, epochs=100, verbose=1)

    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.show()
    scores = model.evaluate(trainX, trainy, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    model.save("personality_model.h5")
    print("Model saved to disk")


def predict_model(model_type, input_data, le):
    model = load_model(model_type + '.h5')
    print(model.summary())
    test_input = np.reshape(input_data, (1, 10))
    test_output = model.predict(test_input, verbose=1)
    output_data = np.argmax(test_output, 1)
    output_class = le.inverse_transform(output_data)
    print("Predicted Output :", output_class[0])


if __name__ == "__main__":
    # for model training comment new_input_array and predict_model fields
    # for model prediction comment each model

    trainX, trainy, testX, testy, le = prepare_data("Learning-Style")
    # trainX, trainy, testX, testy, le = prepare_data("Personality-Trait")
    learning_model(trainX, trainy, testX, testy)
    # learning_model(trainX, trainy, testX, testy)
    new_input_array = [1, 0, 4, 2, 2, 1, 4, 0, 1, 3]  # input
    predict_model("learning_model", new_input_array, le)

    trainX, trainy, testX, testy, le = prepare_data("Personality-Trait")
    personality_model(trainX, trainy, testX, testy)
    new_input_array = [3, 3, 3, 1, 4, 2, 3, 4, 2, 4]  # input
    predict_model("personality_model", new_input_array, le)

    print("Finished process.......")