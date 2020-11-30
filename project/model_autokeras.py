import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing import image
import autokeras as ak

x_train=np.load('./saveDATA/train_x.npy')
y_train=np.load('./saveDATA/train_y.npy')
x_test=np.load('./saveDATA/test_x.npy')
y_test=np.load('./saveDATA/test_y.npy')

test_image=image.load_img('./data/predict/20201127_225236_065.jpg', target_size=(200,200))
test_image=image.img_to_array(test_image)
x_predict=np.expand_dims(test_image, axis=0)


clf=ak.ImageClassifier(
    multi_label=True,
    max_trials=2
)

clf.fit(x_train, y_train, epochs=50, batch_size=100)

loss, accuracy=clf.evaluate(x_test, y_test)
predicted_y=clf.predict(x_predict)
predicted_y=np.argmax(predicted_y, axis=-1)

print('loss :', loss)
print('acc :', accuracy)
print('예측라벨 :', predicted_y)

# epochs : 50 => evaluate() : [10.181744575500488, 0.4046296179294586]

'''
max_trials=1 =>
Trial 1 Complete [00h 06m 45s]
val_loss: 1.9475404024124146

Best val_loss So Far: 1.9475404024124146
Total elapsed time: 00h 06m 45s
evaluate() : [10.181744575500488, 0.4046296179294586]
예측라벨 : [2]
'''