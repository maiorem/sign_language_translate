import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.applications import VGG16, ResNet50, Xception, nasnet, InceptionV3
from tensorflow.keras.preprocessing import image

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

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
    class_mode=None,
    shuffle=False 
)


model = VGG16(include_top=False, weights='imagenet',input_shape = (200,200,3))
# model = ResNet50(include_top=False, weights='imagenet',input_shape = (200,200,3))
# model = Xception(include_top=False, weights='imagenet',input_shape = (200,200,3))
# model = InceptionV3(include_top=False, weights='imagenet',input_shape = (200,200,3))

model.summary()

bottleneck_features_train = model.predict_generator(xy_train, 7200)
np.save(open('./saveDATA/bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

bottleneck_features_validation = model.predict_generator(xy_test, 1000)
np.save(open('./saveDATA/bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

test_image=image.load_img('./data/predict/KakaoTalk_20201130_141254575.jpg', target_size=(200,200))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)

bottleneck_features_predict=model.predict_generator(test_image, 1)
np.save(open('./saveDATA/bottleneck_feature_predict.npy', 'wb'), bottleneck_features_predict)

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
