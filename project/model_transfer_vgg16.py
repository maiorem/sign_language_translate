import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

train_data = np.load(open('./saveDATA/bottleneck_features_train.npy', 'rb'))
train_labels = np.array(([i for i in range(10)]) * 720)
train_labels = np.sort(train_labels)

validation_data = np.load(open('./saveDATA/bottleneck_features_validation.npy', 'rb'))
validation_labels = np.array([i for i in range(10)] * 100)
validation_labels = np.sort(validation_labels)

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

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(train_data, train_labels,
          epochs=50,
          batch_size=16,
          validation_data=(validation_data, validation_labels))

loss, accuracy=model.evaluate(validation_data, validation_labels, batch_size=16)
y_predict=model.predict(predict_data)
y_predict=np.argmax(y_predict, axis=-1)

print('loss :', loss)
print('acc :', accuracy)
print('예측 라벨 :', y_predict)
'''
loss : 0.21312393248081207
acc : 0.9789999723434448
'''

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.plot(acc)
plt.plot(val_acc)
plt.plot(loss)
plt.plot(val_loss)

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')

plt.legend(['acc', 'val_acc', 'loss', 'val_loss'])
plt.show()

