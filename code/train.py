from utils import *
from network import *

BATCH_SIZE = 32
TRAIN_IMAGE_COUNT = 1821
INPUT_SIZE = [512, 352]

with tf.device('/device:cpu:0'):
    train_ds, valid_ds = get_train_valid_dataset('C:\\Users\\xiaow\\Documents\\Plant_pathology_2020\\Data\\Train\\',
                                                image_count=TRAIN_IMAGE_COUNT, image_size=INPUT_SIZE,
                                                batch_size=BATCH_SIZE, valid_split_rate=0.2)
    test_ds = get_test_dataset('C:\\Users\\xiaow\\Documents\\Plant_pathology_2020\\Data\\Test\\',
                               image_size=INPUT_SIZE, batch_size=None)

# print(train_ds)
# print(valid_ds)
# print(test_ds)
# # image_batch, label_batch = next(iter(train_ds))
# # print(image_batch.shape, label_batch.shape)
# print(next(iter(test_ds)).shape)
