import numpy as np
from tensorflow.keras.models import load_model

#1. 데이터
x_train=np.load('./saveDATA/train_x.npy')
y_train=np.load('./saveDATA/train_y.npy')
x_test=np.load('./saveDATA/test_x.npy')
y_test=np.load('./saveDATA/test_y.npy')

#2. 모델 + 훈련
model=load_model('./cp/SLTcnn-17-1.5432.hdf5')

#3. 평가
result=model.evaluate(x_test, y_test, batch_size=100)

print('result :', result)
