import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.constraints import max_norm
from keras.optimizers import Adadelta
from sklearn import svm
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report, confusion_matrix
from keras.utils import plot_model
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from skimage.feature import hog, corner_fast
import pickle
import src.config as cfg
import src.data as dat
import cv2


def load_training_data(network_type):
    if network_type == "hog":
        features = np.load(cfg.pathDATAHOG + '\\' + cfg.folders[0] + '\\' + 'features.npy')
        labels = np.load(cfg.pathDATAHOG + '\\' + cfg.folders[0] + '\\' + 'labels.npy')
        return features, labels
    elif network_type == "fast":
        features = np.load(cfg.pathDATAFAST + '\\' + cfg.folders[0] + '\\' + 'features.npy')
        labels = np.load(cfg.pathDATAFAST + '\\' + cfg.folders[0] + '\\' + 'labels.npy')
        return features, labels
    elif network_type == "x":
        features = np.load(cfg.pathDATAXception + "\\" + cfg.folders[0] + "\\" + 'features.npy')
        labels = np.load(cfg.pathDATAXception + "\\" + cfg.folders[0] + "\\" + '/labels.npy')
        combined = list(zip(features, labels))
        np.random.shuffle(combined)
        percentage = 85
        partition = int(len(features) * percentage / 100)
        train = combined[:partition]
        test = combined[partition:]
        return train, test, features, labels


def load_testing_data(network_type):
    if network_type == "hog":
        f1 = np.load(cfg.pathDATAHOG + '\\' + cfg.folders[1] + '\\' + 'features.npy')
        l1 = np.load(cfg.pathDATAHOG + '\\' + cfg.folders[1] + '\\' + 'labels.npy')
        f2 = np.load(cfg.pathDATAHOG + '\\' + cfg.folders[2] + '\\' + 'features.npy')
        l2 = np.load(cfg.pathDATAHOG + '\\' + cfg.folders[2] + '\\' + 'labels.npy')
        return f1, f2, l1, l2
    elif network_type == "fast":
        f1 = np.load(cfg.pathDATAFAST + '\\' + cfg.folders[1] + '\\' + 'features.npy')
        f1_features = np.array(f1).reshape((len(f1), -1))
        l1 = np.load(cfg.pathDATAFAST + '\\' + cfg.folders[1] + '\\' + 'labels.npy')
        f2 = np.load(cfg.pathDATAFAST + '\\' + cfg.folders[2] + '\\' + 'features.npy')
        f2_features = np.array(f2).reshape((len(f2), -1))
        l2 = np.load(cfg.pathDATAFAST + '\\' + cfg.folders[2] + '\\' + 'labels.npy')
        return f1_features, f2_features, l1, l2
    elif network_type == "x":
        f1 = np.load(cfg.pathDATAXception + '\\' + cfg.folders[1] + '\\' + 'features.npy')
        l1 = np.load(cfg.pathDATAXception + '\\' + cfg.folders[1] + '\\' + 'labels.npy')
        f2 = np.load(cfg.pathDATAXception + '\\' + cfg.folders[2] + '\\' + 'features.npy')
        l2 = np.load(cfg.pathDATAXception + '\\' + cfg.folders[2] + '\\' + 'labels.npy')
        return f1, f2, l1, l2


def process_data_learn(data_type, fast, features, labels):
    if data_type == "train":
        f_features = np.array(features)
        f_labels = np.array(labels).reshape(len(labels), 1)
        if fast == "yes":
            f_features = np.array(f_features).reshape((len(f_features), -1))
        data_frame = np.hstack((f_features, f_labels))
        np.random.shuffle(data_frame)
        percentage = 80
        partition = int(len(f_features) * percentage / 100)
        x_train, x_test = data_frame[:partition, :-1], data_frame[partition:, :-1]
        y_train, y_test = data_frame[:partition, -1:].ravel(), data_frame[partition:, -1:].ravel()
        return x_train, x_test, y_train, y_test
    elif data_type == "test":
        f_features = np.array(features).reshape((len(features), -1))
        f_labels = np.array(labels).reshape(len(labels), 1)
        if fast == "yes":
            f_features = np.array(f_features).reshape((len(f_features), -1))
        data_frame = np.hstack((f_features, f_labels))
        np.random.shuffle(data_frame)
        percentage = 80
        partition = int(len(f_features) * percentage / 100)
        x_train, x_test = data_frame[:partition, :-1], data_frame[partition:, :-1]
        y_train, y_test = data_frame[:partition, -1:].ravel(), data_frame[partition:, -1:].ravel()
        return x_train, x_test, y_train, y_test


def train_svm_hog():
    filename = 'hog_model'
    features, labels = load_training_data("hog")
    x_train, x_test, y_train, y_test = process_data_learn("train", "no", features, labels)
    model = svm.SVC(verbose=True)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    plot = plot_confusion_matrix(model, x_test, y_test, display_labels=cfg.label_names, cmap=plt.cm.Blues,
                                 normalize='true')
    plot.ax_.set_title("Confusion matrix SMV+HOG")
    plt.savefig(cfg.pathModelsHOG + "\\" + filename + ".png")
    f1, f2, l1, l2 = load_testing_data("hog")
    test1 = model.predict(f1)
    test2 = model.predict(f2)
    a = "Train Accuracy: " + str(round(accuracy_score(y_test, prediction) * 100, 4)) + "%"
    b = "Private Test Accuracy: " + str(round(accuracy_score(l1, test1) * 100, 4)) + "%"
    c = "Public Test Accuracy: " + str(round(accuracy_score(l2, test2) * 100, 4)) + "%"
    file = open(cfg.pathModelsHOG + "\\" + filename + ".txt", "w")
    file.writelines([a + "\n", b + "\n", c + "\n"])
    file.close()
    print(a + "\n", b + "\n", c + "\n")
    pickle.dump(model, open(cfg.pathModelsHOG + "\\" + filename + ".sav", 'wb'))


def load_hog_model():
    filename = 'hog_model'
    loaded_model = pickle.load(open(cfg.pathModelsHOG + "\\" + filename + ".sav", 'rb'))
    file = open(cfg.pathModelsHOG + "\\" + filename + ".txt", "r")
    line = file.readlines()
    file.close()
    accuracy = ""
    for li in line:
        accuracy = accuracy + "%s" % li
    print(accuracy)
    print("HOG Model loaded successfully\n")
    return loaded_model


def train_svm_fast():
    filename = 'fast_model'
    features, labels = load_training_data("fast")
    x_train, x_test, y_train, y_test = process_data_learn("train", "yes", features, labels)
    model = svm.SVC(verbose=True)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    plot = plot_confusion_matrix(model, x_test, y_test, display_labels=cfg.label_names, cmap=plt.cm.Blues,
                                 normalize='true')
    plot.ax_.set_title("Confusion matrix SMV+FAST")
    plt.savefig(cfg.pathModelsFAST + "\\" + filename + ".png")
    f1, f2, l1, l2 = load_testing_data("fast")
    test1 = model.predict(f1)
    test2 = model.predict(f2)
    a = "Train Accuracy: " + str(round(accuracy_score(y_test, prediction) * 100, 4)) + "%"
    b = "Private Test Accuracy: " + str(round(accuracy_score(l1, test1) * 100, 4)) + "%"
    c = "Public Test Accuracy: " + str(round(accuracy_score(l2, test2) * 100, 4)) + "%"
    file = open(cfg.pathModelsFAST + "\\" + filename + ".txt", "w")
    file.writelines([a + "\n", b + "\n", c + "\n"])
    file.close()
    print(a + "\n", b + "\n", c + "\n")
    pickle.dump(model, open(cfg.pathModelsFAST + "\\" + filename + ".sav", 'wb'))


def load_fast_model():
    filename = 'fast_model'
    loaded_model = pickle.load(open(cfg.pathModelsFAST + "\\" + filename + ".sav", 'rb'))
    file = open(cfg.pathModelsFAST + "\\" + filename + ".txt", "r")
    line = file.readlines()
    file.close()
    accuracy = ""
    for li in line:
        accuracy = accuracy + "%s" % li
    print(accuracy)
    print("FAST Model loaded successfully\n")
    return loaded_model


def train_xception():
    filename = 'x_model'
    train, test, features, labels = load_training_data("x")
    f1, f2, l1, l2 = load_testing_data("x")
    model = get_model(cfg.label_names)
    checkpoint = ModelCheckpoint(cfg.pathModelsXception + "/best_model.h5", monitor="val_accuracy", verbose=1,
                                 save_best_only=True, mode='max', period=1)
    model.fit_generator(
        generator=dat.DataGenerator(train),
        steps_per_epoch=int(len(train)/128),
        epochs=1,
        validation_data=dat.DataGenerator(test),
        validation_steps=int(len(test)/128),
        workers=0,
        callbacks=[checkpoint]
    )
    a = model.predict(features)
    a = np.argmax(a, axis=1)
    lab = np.argmax(labels, axis=1)
    report = confusion_matrix(lab, a, normalize='true')
    print(report)
    model.save(cfg.pathModelsXception + "\\" + filename + ".h5")
    loss, acc = model.evaluate(features, labels)
    loss1, acc1 = model.evaluate(f1, l1)
    loss2, acc2 = model.evaluate(f2, l2)
    a = "Private Test Accuracy: " + str(round(acc * 100, 4)) + "%"
    b = "Private Test Accuracy: " + str(round(acc1 * 100, 4)) + "%"
    c = "Public Test Accuracy: " + str(round(acc2 * 100, 4)) + "%"
    file = open(cfg.pathModelsXception + "\\" + filename + ".txt", "w")
    file.writelines([a + "\n", b + "\n", c + "\n"])
    file.close()
    print(a + "\n", b + "\n", c + "\n")


def create_x_files():
    filename = 'x_model'
    train, test, features, labels = load_training_data("x")
    f1, f2, l1, l2 = load_testing_data("x")
    model = load_model(cfg.pathModelsXception + "\\" + filename + ".h5")
    a = model.predict(features)
    a = np.argmax(a, axis=1)
    lab = np.argmax(labels, axis=1)
    report = confusion_matrix(lab, a, normalize='true')
    print(report)
    model.save(cfg.pathModelsXception + "\\" + filename + ".h5")
    loss, acc = model.evaluate(features, labels)
    loss1, acc1 = model.evaluate(f1, l1)
    loss2, acc2 = model.evaluate(f2, l2)
    a = "Private Test Accuracy: " + str(round(acc * 100, 4)) + "%"
    b = "Private Test Accuracy: " + str(round(acc1 * 100, 4)) + "%"
    c = "Public Test Accuracy: " + str(round(acc2 * 100, 4)) + "%"
    file = open(cfg.pathModelsXception + "\\" + filename + ".txt", "w")
    file.writelines([a + "\n", b + "\n", c + "\n"])
    file.close()
    print(a + "\n", b + "\n", c + "\n")


def load_xception():
    filename = 'x_model'
    loaded_model = load_model(cfg.pathModelsXception + "\\" + filename + ".h5")
    file = open(cfg.pathModelsXception + "\\" + filename + ".txt", "r")
    line = file.readlines()
    file.close()
    accuracy = ""
    for li in line:
        accuracy = accuracy + "%s" % li
    print(accuracy)
    print("Xception Model loaded successfully\n")
    return loaded_model


def get_model(all_labels):
    model_base = keras.applications.xception.Xception(include_top=False, input_shape=(*(71, 71), 3), weights='imagenet')
    output = Flatten()(model_base.output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(128, activation='relu', kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(len(all_labels), activation='softmax', kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(
        output)
    model = Model(model_base.input, output)
    for layer in model_base.layers:
        layer.trainable = True
    model.summary(line_length=200)
    plot_model(model, show_shapes=True, to_file=cfg.outputPathLogsXception + "\\Xception.pdf")
    ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(optimizer=ada,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def do_prediction(path, network_type, network_model):
    entry = []
    if network_type == "hog":
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
        final_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        entry.append(final_image)
        result = network_model.predict(entry)
        return str("Network using HOG - " + path + " - Result: " + cfg.Emotion(int(result.item(0))).name)
    elif network_type == "fast":
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
        final_image = corner_fast(image)
        entry.append(final_image)
        entry1 = np.array(entry).reshape((len(entry), -1))
        result = network_model.predict(entry1)
        return str("Network using FAST - " + path + " - Result: " + cfg.Emotion(int(result.item(0))).name)
    elif network_type == "x":
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (71, 71), interpolation=cv2.INTER_AREA)
        entry.append(image)
        result = network_model.predict(np.asarray(entry))
        return str("Network using Xception - " + path + " - Result: " + cfg.Emotion(
            np.argmax(result, axis=None, out=None)).name)
