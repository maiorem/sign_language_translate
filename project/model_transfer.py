import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

train_data = np.load(open('bottleneck_features_train.npy'))
# 앞서 언급한 바와 같이 병목 특징은 순서대로 추출되기 때문에 라벨 데이터는 아래와 같이 손쉽게 생성할 수 있습니다.
train_labels = np.array([0] * 1000 + [1] * 1000)

validation_data = np.load(open('bottleneck_features_validation.npy'))
validation_labels = np.array([0] * 400 + [1] * 400)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=50,
          batch_size=16,
          validation_data=(validation_data, validation_labels))
model.save_weights('./save_weights/bottleneck_fc_model.h5')
