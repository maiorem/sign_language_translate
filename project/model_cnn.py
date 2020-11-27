import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


x_train=np.load('./saveDATA/train_x.npy')
y_train=np.load('./saveDATA/train_y.npy')
x_test=np.load('./saveDATA/test_x.npy')
y_test=np.load('./saveDATA/test_y.npy')

y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)

model=Sequential()
model.add(Conv2D(50, (2,2), input_shape=(150,150,3)))
model.add(Conv2D(100, (3,3)))
model.add(Conv2D(70, (2,2)))
model.add(Conv2D(50, (2,2)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es=EarlyStopping(monitor='val_loss',  patience=100, mode='auto')
modelpath='./cp/SLTcnn-{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=10000, batch_size=100, validation_split=0.2, callbacks=[es, cp])

loss, acc=model.evaluate(x_test, y_test, batch_size=100)

print('loss:' ,loss)
print('acc :', acc)