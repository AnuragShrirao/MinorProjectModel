from joblib import dump, load
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("dataset.csv")
# print(data.keys())
clf = DecisionTreeClassifier()
clf.fit(data.drop(columns = ["Name ","Roll number","placement"]),data["placement"])
dump(clf, "model.ml")























# clf = LogisticRegression()
# X = clf.iloc[:, 2:5].values
# y = clf.iloc[:, 6].values
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.layers import Input, Dense, Activation,Dropout
# from tensorflow.keras.models import Model

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# input_layer = Input(shape=(X.shape[1],))
# dense_layer_1 = Dense(100, activation='relu')(input_layer)
# dense_layer_2 = Dense(50, activation='relu')(dense_layer_1)
# dense_layer_3 = Dense(25, activation='relu')(dense_layer_2)
# output = Dense(1)(dense_layer_3)

# model = Model(inputs=input_layer, outputs=output)
# model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])


# tf_model = tf.keras.models.Sequential()
# tf_model.add(tf.keras.Input(shape=(21,)))
# tf_model.add(tf.keras.layers.Dense(1))

# # assign the parameters from sklearn to the TF model
# tf_model.layers[0].weights[0].assign(clf.coef_.transpose())
# tf_model.layers[0].bias.assign(clf.intercept_)

# # verify the models do the same prediction
# assert np.all((tf_model(x) > 0)[:, 0].numpy() == clf.predict(x))



# import tensorflow as tf
# import numpy as np
# from sklearn.linear_model import LogisticRegression

# # some random data to train and test on
# x = np.random.normal(size=(60, 21))
# y = np.random.uniform(size=(60,)) > 0.5

# # fit the sklearn model on the data
# sklearn_model = LogisticRegression().fit(x, y)

# # create a TF model with the same architecture
# tf_model = tf.keras.models.Sequential()
# tf_model.add(tf.keras.Input(shape=(21,)))
# tf_model.add(tf.keras.layers.Dense(1))

# # assign the parameters from sklearn to the TF model
# tf_model.layers[0].weights[0].assign(sklearn_model.coef_.transpose())
# tf_model.layers[0].bias.assign(sklearn_model.intercept_)

# # verify the models do the same prediction
# assert np.all((tf_model(x) > 0)[:, 0].numpy() == sklearn_model.predict(x))