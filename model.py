import mahotas as mh
from matplotlib import pyplot as plt
from glob import glob
import os
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.linear_model import LogisticRegression

IMM_SIZE = 224


def get_data(folder):
    # ctreate a list of SubFolders
    class_names = [f for f in os.listdir(folder) if not f.startswith('.')]
    data = []
    print(class_names)
    for t, f in enumerate(class_names):
        images = glob(folder + "/" + f + "/*")  # create a list of files
        print("Downloading: ", f)
        fig = plt.figure(figsize=(50, 50))
        for im_n, im in enumerate(images):
            plt.gray()  # set grey colormap of images
            image = mh.imread(im)
            if len(image.shape) > 2:
                # resize of RGB and png images
                image = mh.resize_to(
                    image, [IMM_SIZE, IMM_SIZE, image.shape[2]])
            else:
                # resize of grey images
                image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE])
            if len(image.shape) > 2:
                # change of colormap of images alpha chanel delete
                image = mh.colors.rgb2grey(image[:, :, :3], dtype=np.uint8)
            # create a table of images
            plt.subplot(int(len(images)/5)+1, 5, im_n+1)
            plt.imshow(image)
            data.append([image, f])
        plt.show()

    return np.array(data)


trainD = "/Covid19-dataset/train"
train = get_data(trainD)

testD = "/Covid19-dataset/test"
val = get_data(testD)

print("Train shape", train.shape)  # Size of the training DataSet
print("Test shape", val.shape)  # Size of the test DataSet
print("Image size", train[0][0].shape)  # Size of image


def create_features(data):
    features = []
    labels = []
    for image, label in data:
        features.append(mh.features.haralick(image).ravel())
        labels.append(label)
    features = np.array(features)
    labels = np.array(labels)
    return (features, labels)


features_train, labels_train = create_features(train)
features_test, labels_test = create_features(val)

clf = Pipeline([('preproc', StandardScaler()),
               ('classifier', LogisticRegression())])
clf.fit(features_train, labels_train)

score = clf.score(features_test, labels_test)
pred = clf.predict(features_test, labels_test)

print('Test DataSet accuracy: {: .1%}'.format(score))

filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))
