import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.applications.vgg16 import VGG16

train_datagen = ImageDataGenerator(rescale=1./255, 
                                horizontal_flip=True,
                                vertical_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                rotation_range=40,
                                zoom_range=0.2,
                                shear_range=0.2,
                                fill_mode='nearest'
                                )
test_datagen = ImageDataGenerator(rescale=1./255)
# validation_datagen=ImageDataGenerator(rescale=1./255)

xy_train=train_datagen.flow_from_directory(
    './data/train',
    target_size=(200,200),
    batch_size=16,
    class_mode=None,
    shuffle=False 
) 

xy_test=test_datagen.flow_from_directory(
    './data/test',
    target_size=(200,200),
    batch_size=16,
    class_mode='categorical' 
)


model = VGG16(include_top=False, weights='imagenet',input_shape = (200,200,3))


# 이미지를 모델에 입력시켜 결과를 가져옵니다. 본래 어떤 예측 결과가 출력되어야 하지만 모델의 일부만 가져왔기 때문에 병목 특징이 출력됩니다.
bottleneck_features_train = model.predict_generator(xy_train, 7200)
# 출력된 병목 데이터를 저장합니다.
np.save(open('./saveDATA/bottleneck_features_train.npy', 'w'), bottleneck_features_train)

bottleneck_features_validation = model.predict_generator(xy_test, 1000)
np.save(open('./saveDATA/bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# history=model.fit_generator(
#     xy_train,
#     steps_per_epoch=1000 // 16,
#     epochs=50,
#     validation_data=xy_test,
#     validation_steps=4
# )
# model.save_weights('./save_weights/fit_gen.h5')

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
