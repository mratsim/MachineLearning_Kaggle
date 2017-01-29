import numpy as np
import pandas as pd

train = pd.read_csv('./data/train.csv', delimiter=',', header=0)
test = pd.read_csv('./data/test.csv', delimiter=',', header=0)

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

from keras.layers import Input, merge, Flatten, Activation, Reshape
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

from keras.callbacks import History, ModelCheckpoint, CSVLogger, EarlyStopping
from keras.optimizers import Adam

import time


# Keras neural Network

# Get dimensions
nb_dim = train.shape[1] - 1 # 784 features
nb_classes = pd.get_dummies(train['label']).shape[1] #10 possible values for label
img_2Dshape = (1,28, 28)

# Callbacks
history = History()
checkpointer = ModelCheckpoint(filepath=time.strftime("%Y-%m-%d_%H%M-")+"DCNN_weights.hdf5",
                               verbose=1, save_best_only=False)
csv_logger = CSVLogger(time.strftime("%Y-%m-%d_%H%M-")+'training.log')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto')


# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential([
        Reshape(img_2Dshape, input_shape=(nb_dim,)),
        
        ZeroPadding2D(padding=(3, 3)),
        Convolution2D(16,7,7,init='he_normal'),
        Activation('relu'),
        BatchNormalization(),
        ZeroPadding2D(padding=(2, 2)),
        Convolution2D(32,5,5,init='he_normal'),
        Activation('relu'),
        BatchNormalization(),
        
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), #Post-Size 14x14

        ZeroPadding2D(padding=(2, 2)),
        Convolution2D(64,5,5,init='he_normal'),
        Activation('relu'),
        BatchNormalization(),
        ZeroPadding2D(padding=(1, 1)),
        Convolution2D(128,3,3,init='he_normal'),
        Activation('relu'),
        BatchNormalization(),
        
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), #Post-Size 7x7
        
        Convolution2D(4096,7,7,init='he_normal'),
        Activation('relu'),
        Convolution2D(nb_classes,1,1,init='he_normal'),
        Flatten(),
        Activation('softmax')
    ])
    
   
    # Compile model
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model

clf_keras = KerasClassifier(build_fn=create_model, nb_epoch=256, batch_size=256, verbose=1,
                            validation_split=0.1,
                            callbacks=[history,checkpointer,csv_logger,early_stop])

# fix random seed for reproducibility
seed = 777
np.random.seed(seed)

featurelist = (''.join(['pixel',str(i)]) for i in range(784))
#featurelist generator

mapper = DataFrameMapper(
    [ ([pixel], None) for pixel in featurelist]
)
# Do we need StandardScaler() ? It's the same scale of color for all pixels

pipe = Pipeline([
    ("featurize", mapper),
    ("mlp", clf_keras)
])

########################Helper functions ################
##### Cross Validation
def crossval():
    # split = ShuffleSplit(n_splits=10) #compat issue with sklearn pandas "object is not iterable"
    cv = cross_val_score(pipe, X_train, y_train, cv=10, n_jobs=1) #Note : don't use parallel crossval  and GPU computation
    print("Cross Validation Scores are: ", cv.round(3))
    print("Mean CrossVal score is: ", round(cv.mean(),3))
    print("Std Dev CrossVal score is: ", round(cv.std(),3))
    return cv
def output():
    predictions = pipe.predict(test)
    result = pd.DataFrame(predictions)
    result.index+=1
    result.index.names=['ImageId']
    result.columns=['Label']
    result.to_csv(time.strftime("%Y-%m-%d_%H%M-") + 'keras.csv', header=True)

X_train = train
y_train = train['label']

#temp_cv = crossval()

pipe.fit(X_train, y_train)

output()