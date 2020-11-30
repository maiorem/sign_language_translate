import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from sklearn.ensemble import RandomForestClassifier

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

#PCA로 컬럼 걸러내기
pca=PCA()
pca.fit(x)
cumsum=np.cumsum(pca.explained_variance_ratio_) #누적된 합 표시
# print(cumsum)

d=np.argmax(cumsum >= 0.95) + 1
# print(cumsum>=0.95) 
print(d) # 154

pca1=PCA(n_components=d)
x=pca1.fit_transform(x)
x_predict=pca1.fit_transform(x_predict)

print(x.shape) #(70000, 154)
print(x_predict.shape)

x_train=x[:7200, :]
x_test=x[7200:, :]


#2. 모델
model=RandomForestClassifier()

#3.훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score=model.score(x_test, y_test)

y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=-1)

print('score : ', score)
print('예측 라벨 :', y_predict)


