import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

x_train=np.load('./saveDATA/train_x.npy')
y_train=np.load('./saveDATA/train_y.npy')
x_test=np.load('./saveDATA/test_x.npy')
y_test=np.load('./saveDATA/test_y.npy')

test_image=image.load_img('./data/predict/20201127_225236_065.jpg', target_size=(200,200))
test_image=image.img_to_array(test_image)
x_predict=np.expand_dims(test_image, axis=0)

model=Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(200,200,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es=EarlyStopping(monitor='val_loss',  patience=100, mode='auto')
modelpath='./cp/SLTcnn-{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
history=model.fit(x_train, y_train, epochs=10000, batch_size=100, validation_split=0.3, callbacks=[es, cp])

loss, acc=model.evaluate(x_test, y_test, batch_size=100)

y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=-1)
print('loss:' ,loss)
print('acc :', acc)
print('예측 라벨 : ', y_predict)

'''
loss: 0.5933569073677063
acc : 0.8989999890327454
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
