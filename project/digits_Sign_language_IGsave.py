import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing import image

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
    class_mode='categorical' 
) 

xy_test=test_datagen.flow_from_directory(
    './data/test',
    target_size=(200,200),
    batch_size=16,
    class_mode='categorical' 
)
# validation_generator = validation_datagen.flow_from_directory(
#         './data/validation',
#         target_size=(150, 150),
#         batch_size=16,
#         class_mode='categorical'
# )

# # 이미지 증폭 디버깅
# img = load_img('./data/train/0/four sign fingers hand90.jpg') 
# x = img_to_array(img)  
# x = x.reshape((1,) + x.shape) 

# i = 0
# for batch in train_datagen.flow(x, 
#                             batch_size=1,
#                             save_to_dir='./preview', 
#                             save_prefix='0', 
#                             save_format='jpg'):
#     i += 1
#     if i > 20:
#         break 

# print(xy_train[0][0].shape) #(7200, 200, 200, 3)
# print(xy_train[0][1].shape) #(7200, 10)
# print(xy_test[0][0].shape) #(1000, 200, 200, 3)
# print(xy_test[0][1].shape) #(1000, 10)

# np.save('./saveDATA/train_x.npy', arr=xy_train[0][0])
# np.save('./saveDATA/train_y.npy', arr=xy_train[0][1])
# np.save('./saveDATA/test_x.npy', arr=xy_test[0][0])
# np.save('./saveDATA/test_y.npy', arr=xy_test[0][1])


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

history=model.fit_generator(
    xy_train,
    steps_per_epoch=1000 // 16,
    epochs=50,
    validation_data=xy_test,
    validation_steps=4
)
model.save_weights('./save_weights/fit_gen.h5')

test_image=image.load_img('./data/predict/20201127_225159_040.jpg', target_size=(200,200))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)

result=model.predict(test_image)
print(result)
print(result[0][0])
np.save('./saveDATA/predict_image.npy', arr=test_image)

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
