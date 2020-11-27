import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak

x_train=np.load('./saveDATA/train_x.npy')
y_train=np.load('./saveDATA/train_y.npy')
x_test=np.load('./saveDATA/test_x.npy')
y_test=np.load('./saveDATA/test_y.npy')

clf=ak.ImageClassifier(
    overwrite=True,
    max_trials=2
)

clf.fit(x_train, y_train, epochs=50)

# x_predict=np.load()
# predicted_y=clf.predict(x_predict)

# print('predicted_y :', predicted_y)
print('evaluate() :', clf.evaluate(x_test, y_test))