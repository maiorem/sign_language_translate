import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing import image

x_train=np.load('./saveDATA/train_x.npy')
y_train=np.load('./saveDATA/train_y.npy')
x_test=np.load('./saveDATA/test_x.npy')
y_test=np.load('./saveDATA/test_y.npy')

test_image=image.load_img('./data/predict/20201127_225236_065.jpg', target_size=(200,200))
test_image=image.img_to_array(test_image)
x_predict=np.expand_dims(test_image, axis=0)

x=np.append(x_train, x_test, axis=0)
x=x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
x_predict=x_predict.reshape(x_predict.shape[0], x_predict.shape[1]*x_predict.shape[2]*x_predict.shape[3])

print(x.shape) #(8200, 120000)
print(x_predict.shape) #(1, 120000)

pca1=PCA(n_components=0.95)
x=pca1.fit_transform(x)

print(x.shape) #(8200, 345)


x_train=x[:7200, :]
x_test=x[7200:, :]

print(x_train.shape)
print(y_train.shape)



#2. 모델
model=Sequential()
model.add(Dense(2000, activation='relu', input_shape=(x.shape[1],)))
model.add(Dense(4000, activation='relu'))
model.add(Dense(3000, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax')) 

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', patience=50, mode='auto')
model.fit(x_train, y_train, epochs=10000, batch_size=1000, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=1000)


print('loss : ', loss)
print('accuracy : ', accuracy)


