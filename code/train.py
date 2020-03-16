from utils import *
from network import *

BATCH_SIZE = 32
TRAIN_IMAGE_COUNT = 1821
INPUT_SHAPE = [512, 352, 3]

with tf.device('/cpu:0'):
    train_ds, valid_ds = get_train_valid_dataset('C:\\Users\\xiaow\\Documents\\Plant_pathology_2020\\Data\\Train\\',
                                                image_count=TRAIN_IMAGE_COUNT, image_size=INPUT_SHAPE[0:2],
                                                batch_size=BATCH_SIZE, valid_split_rate=0.2)
    test_ds = get_test_dataset('C:\\Users\\xiaow\\Documents\\Plant_pathology_2020\\Data\\Test\\',
                               image_size=INPUT_SHAPE[0:2], batch_size=None)

# print(train_ds)
# print(valid_ds)
# print(test_ds)
# image_batch, label_batch = next(iter(train_ds))
# print(image_batch.shape, label_batch.shape)
# print(next(iter(test_ds)).shape)

# model = fake_alexnet_model(INPUT_SHAPE)
model = normal_block_classifier(INPUT_SHAPE, levels=5, alpha=0.3, init_channels=32, dropout_rate=0)
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
model.summary()

'''
log_dir="training_logs\\" + "20200320_normalblock_512x352_c32_level5_alpha0.3"
tensorboard_callback = TensorBoard(
    log_dir=log_dir, update_freq=20, histogram_freq=1)
    
model.fit(train_ds, epochs=5, verbose=2, validation_data=valid_ds,
          callbacks=[tensorboard_callback])

model.save('20200320_normalblock_512x352_c32_level5_alpha0.3.h5')
'''