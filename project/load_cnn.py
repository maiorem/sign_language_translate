import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#1. 데이터
x_train=np.load('./saveDATA/train_x.npy')
y_train=np.load('./saveDATA/train_y.npy')
x_test=np.load('./saveDATA/test_x.npy')
y_test=np.load('./saveDATA/test_y.npy')
 
test_image=image.load_img('./data/predict/KakaoTalk_20201130_223838600.jpg', target_size=(200,200))
test_image=image.img_to_array(test_image)
x_predict=np.expand_dims(test_image, axis=0)

#2. 모델 + 훈련
model=load_model('./cp/SLTcnn-19-0.7008.hdf5')

#3. 평가
result=model.evaluate(x_test, y_test, batch_size=100)
y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=-1)

print('result :', result)
print('예측 라벨 : ', y_predict)

