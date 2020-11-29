import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


x_train=np.load('./saveDATA/train_x.npy')
y_train=np.load('./saveDATA/train_y.npy')
x_test=np.load('./saveDATA/test_x.npy')
y_test=np.load('./saveDATA/test_y.npy')
x_predict=np.load('./saveDATA/predict_image.npy')

model=Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(200,200,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


es=EarlyStopping(monitor='val_loss',  patience=50, mode='auto')
modelpath='./cp/SLTcnn-{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=10000, batch_size=100, validation_split=0.3, callbacks=[es, cp])

loss, acc=model.evaluate(x_test, y_test, batch_size=100)

y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=-1)
print('loss:' ,loss)
print('acc :', acc)
print('예측 라벨 : ', y_predict)

'''
loss: 3.555645227432251
acc : 0.5659999847412109
'''