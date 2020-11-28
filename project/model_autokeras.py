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

clf.fit(x_train, y_train, epochs=50, batch_size=100)

# x_predict=np.load()
# predicted_y=clf.predict(x_predict)

# print('predicted_y :', predicted_y)
print('evaluate() :', clf.evaluate(x_test, y_test))

# epochs : 50 => evaluate() : [10.181744575500488, 0.4046296179294586]

'''
max_trials=1 =>
Trial 1 Complete [00h 06m 45s]
val_loss: 1.9475404024124146

Best val_loss So Far: 1.9475404024124146
Total elapsed time: 00h 06m 45s
evaluate() : [10.181744575500488, 0.4046296179294586]

max_trials=2 =>
Trial 2 Complete [00h 12m 45s]
val_loss: 2.004319429397583

Best val_loss So Far: 1.9004414081573486
Total elapsed time: 00h 19m 40s
'''