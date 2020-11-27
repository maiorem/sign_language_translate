import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


xy_train=train_datagen.flow_from_directory(
    './data/train',
    target_size=(150,150),
    batch_size=5,
    class_mode='categorical' 
) 

xy_test=test_datagen.flow_from_directory(
    './data/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='categorical' 
)


print(xy_train[0][0].shape)
print(xy_train[0][1].shape)

# np.save('./saveDATA/train_x.npy', arr=xy_train[0][0])
# np.save('./saveDATA/train_y.npy', arr=xy_train[0][1])
# np.save('./saveDATA/test_x.npy', arr=xy_test[0][0])
# np.save('./saveDATA/test_y.npy', arr=xy_test[0][1])