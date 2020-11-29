import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

train_data = np.load(open('./saveDATA/bottleneck_features_train.npy', 'rb'))
train_labels = np.array([0] * 720 + [1] * 720 + [2] * 720 + [3] * 720 + [4] * 720 + [5] * 720 + [6] * 720 + [7] * 720 + [8] * 720 + [9] * 720)

validation_data = np.load(open('./saveDATA/bottleneck_features_validation.npy', 'rb'))
validation_labels = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100 + [5] * 100 + [6] * 100 + [7] * 100 + [8] * 100 + [9] * 100)

predict_data=np.load(open('./saveDATA/bottleneck_feature_predict.npy', 'rb'))

print(train_data.shape) #(7200, 6, 6, 512)
print(train_labels.shape) #(7200,)
print(predict_data.shape) #(1, 6, 6, 512)

train_labels=to_categorical(train_labels) 
validation_labels=to_categorical(validation_labels)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(train_data, train_labels,
          epochs=50,
          batch_size=16,
          validation_data=(validation_data, validation_labels))
model.save_weights('./save_weights/bottleneck_fc_model.h5')

loss, accuracy=model.evaluate(validation_data, validation_labels, batch_size=16)

y_predict=model.predict(predict_data)

y_predict=np.argmax(y_predict, axis=-1)

print('loss :', loss)
print('acc :', accuracy)
print('예측 라벨 :', y_predict)
'''
loss : 0.3532889783382416
acc : 0.9779999852180481
'''

# acc=history.history['accuracy']
# val_acc=history.history['val_accuracy']
# loss=history.history['loss']
# val_loss=history.history['val_loss']

# plt.plot(acc)
# plt.plot(val_acc)
# plt.plot(loss)
# plt.plot(val_loss)

# plt.title('loss & acc')
# plt.ylabel('loss, acc')
# plt.xlabel('epoch')

# plt.legend(['loss', 'val_loss', 'acc', 'val_acc'])
# plt.show()

