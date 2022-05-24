# This demo requires the manual download of the cityscapes-datset from https://www.cityscapes-dataset.com/
# you will need to download gtFine_train_val_test and LeftImg8bit_train_val_test Zip files
# we only use 20 classes of the 34, by creating labelTrainids with createTrainIdLabelImgs.py script
# from https://github.com/mcordts/cityscapesScripts
# Furthermore, this demo uses a local gpu with Cuda enabled.
# disclaimer: some auxillary functions are based on https://github.com/lexfridman/mit-deep-learning/blob/master/tutorial_driving_scene_segmentation/tutorial_driving_scene_segmentation.ipynb

# STEP 1: import required modules
import os
from glob import glob

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import tensorflow as tf
import segmentation_models as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

# for gpu stability
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

# STEP 2: define constants
AUTOTUNE = tf.data.experimental.AUTOTUNE
# Set image height and width. Height 1024 and width 2048 is original resolution
IMAGE_HEIGHT = 96  # for demonstration purposes
IMAGE_WIDTH = 192  # keep original aspect ratio
# 18 432 pixels per channel for (96,192)
# original size: 2 097 152 pixels !! = 114x our size
IMAGE_CHANNELS = 3  # RGB
BATCH_SIZE = 8  # max batch_size i can run on laptop for this resolution
N_CLASSES = 20
LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle', 'void'])

# define path to image and masks (= rootdir)
PATH = r'C:\Users\rembrandt.deville\PycharmProjects\cityscapes_tensorflow'

# seeding for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# STEP 3: load data
# load list of paths to images and masks

train_image_path = sorted(glob(os.path.join(PATH, r'leftImg8bit_trainvaltest\leftImg8bit\train\*.png')))
train_mask_path = sorted(glob(os.path.join(PATH, r'gtFine_trainvaltest\gtFine\train\*\*_labelTrainIds.png')))
test_image_path = sorted(glob(os.path.join(PATH, r'leftImg8bit_trainvaltest\leftImg8bit\val\*.png')))
test_mask_path = sorted(glob(os.path.join(PATH, r'gtFine_trainvaltest\gtFine\val\*\*_labelTrainIds.png')))

# Because we have not much data, we throw all the data together and split 80:10:10
image_path = train_image_path + test_image_path
mask_path = train_mask_path + test_mask_path
# Split in 80% train, 20% validation
train_images, valid_images = train_test_split(image_path, test_size=0.2, random_state=42)
# split the validation half validation and half test
valid_images, test_images = train_test_split(valid_images, test_size=0.5, random_state=42)
train_masks, valid_masks = train_test_split(mask_path, test_size=0.2, random_state=42)
valid_masks, test_masks = train_test_split(valid_masks, test_size=0.5, random_state=42)

num_train_images = len(train_images)
num_valid_images = len(valid_images)
num_test_images = len(test_images)
print(f"Number of training images: {int(num_train_images)}")
print(f"Number of validation images: {int(num_valid_images)}")
print(f"Number of test images: {int(num_test_images)}")


# STEP 4: Create functions for data handling
def read_image(path):  # Load original image
    image = tf.io.read_file(path)  # read file path
    image = tf.image.decode_png(image, channels=3)  # decode file to RGB image
    image = tf.image.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH),
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # resize from (1024, 2048) to (96, 192)
    image = tf.cast(image, tf.float32)  # change data-type to tf.float32
    return image


def read_mask(path):  # Load semantic segmentation mask
    mask = tf.io.read_file(path)  # read file path
    mask = tf.image.decode_png(mask, channels=1)  # decode file to grayscale image
    mask = tf.image.resize(mask, (IMAGE_HEIGHT, IMAGE_WIDTH),
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # resize from (1024, 2048) to (96, 192)
    mask = tf.cast(mask, tf.float32)  # change data-type to tf.float32
    return mask


def image_handler_train(image_path, mask_path):  # image, mask handler for tf.map function
    image = read_image(image_path)  # read the images
    mask = read_mask(mask_path)  # read the masks
    # normalize the image if it is not done automatically in the model (uncomment the right scaling):
    image = image / 127.5 - 1  # for [-1,1] e.g. mobilenetv2
    # image = image/ 255    # for [0,1]
    # image = image # for [0,255] e.g. efficientnet

    # if you want you could also do data augmentation here, but remember to make a separate
    # handler for test set, because those images don't need augmentation
    return image, mask


def mask_to_categorical(image, mask):  # one-hot encode handler for tf.map
    mask = tf.one_hot(tf.cast(mask, tf.int32), N_CLASSES, axis=-1)  # add one-hot coded masks to last dimension
    mask = tf.cast(mask, tf.float32)  # loss functions expects float32
    mask = tf.squeeze(mask, axis=2)  # go from [img_height, img_width, 1, one-hot] to [img_height, img_width, one-hot]
    return image, mask


# STEP 5: initiate training dataset to calculate class weights
# create a datset with tf.data.dataset API for fast tensor handling
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
# read the images and masks into a dataset in parallel
train_dataset = train_dataset.shuffle(len(train_images)).map(image_handler_train, num_parallel_calls=AUTOTUNE)

# let's calculate the class weights before we one-hot encode the masks
# extract and concatenate all the masks of the training set
y = np.concatenate([y for x, y in train_dataset], axis=0)
# count the unique labels and their frequencies
(unique, counts) = np.unique(y, return_counts=True)
class_weights = []
# calculate the class_weights
for i in range(N_CLASSES):
    # class weight for class i = total pixels / n_pixels of j * n_classes
    if counts[i] != 0:
        class_weights.append(sum(counts) / (counts[i] * N_CLASSES))

    else:

        class_weights.append(0)
# normalize the weights to [0,1]
norm_weights = [(class_weights[i] / max(class_weights)) for i in range(len(class_weights))]
# you can manually set weights to zero if you don't want a certain class to be learned by the model
# Try custom weights: gave more weight to road, car, person, traffic pole, traffic sign, sidewalk, rider
custom_weights = [1, 1, 1, 0.1, 0.1, 1,
                  1, 1, 1, 1, 0.1, 1,
                  1, 1, 1, 1, 0.1, 0.1,
                  1, 0]

# create dictionaries of the weight for quick check
norm_weights_dict = {LABEL_NAMES[i]: norm_weights[i] for i in range(N_CLASSES)}
custom_weights_dict = {LABEL_NAMES[i]: custom_weights[i] for i in range(N_CLASSES)}
print(f'These are the normalized class weights: {norm_weights_dict}')
print(f'Thse are the normalized custom weights: {custom_weights_dict}')

# STEP 6: Create training, validation and test set with tf.data.dataset api
# Let's continue with the creation of our datasets:
# Optimize the train_dataset and encode the masks:
# Cache images for further epochs
# Shuffle: 'For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.'
# Repeat to show the training images multiple times
# One-hot encode the masks
train_dataset = train_dataset.cache().repeat(1).map(mask_to_categorical,
                                                    num_parallel_calls=AUTOTUNE)
# further optimization:
# optimize speed with autotune prefetch
# put data in batch with predefined batch size
# you can experiment with ignore_order optimization (not applied here)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE).batch(BATCH_SIZE)

# validation set, same procedure as training set
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_images, valid_masks))
valid_dataset = valid_dataset.shuffle(len(valid_images)).map(image_handler_train, num_parallel_calls=AUTOTUNE)
valid_dataset = valid_dataset.cache().map(mask_to_categorical, num_parallel_calls=AUTOTUNE).prefetch(
    buffer_size=AUTOTUNE).batch(BATCH_SIZE)

# test set, requires no shuffling, prefetching or caching because it is only used for inference
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_path, test_mask_path))
test_dataset = test_dataset.map(image_handler_train, num_parallel_calls=AUTOTUNE).map(mask_to_categorical,
                                                                                      num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# STEP 7: load the model, using Segmentation-models library
# we use mobilenetv2 encoder with pretrained weights on imagenet dataset.
model = sm.Unet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                backbone_name='mobilenetv2',
                encoder_weights='imagenet',
                classes=N_CLASSES,
                activation='softmax',
                encoder_freeze=True)

# let's have a look at the model
model.summary()

# STEP 8: We create callbacks to monitor and change the training process
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.1, patience=7)
save_callback = tf.keras.callbacks.ModelCheckpoint(
    "checkpoint/", save_best_only=True, save_weights_only=True, monitor='val_loss')
earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=15,
                                                          restore_best_weights=True
                                                          )
callbacks = [lr_callback, save_callback, earlystopping_callback]

# STEP 9: define the model.compile parameters
# we use Adam optimizer with default learning_rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# for the loss function we make a custom loss
# weighted_loss = sm.losses.DiceLoss(class_weights=custom_weights)
weighted_loss = sm.losses.JaccardLoss(class_weights=custom_weights)
# uncomment previous line if you want to try Jaccard Loss instead of Dice loss
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = weighted_loss + (1 * focal_loss)

# We track Intersection-over-union and F1score as metrics
metric_iou = sm.metrics.IOUScore()
metric_f1 = sm.metrics.FScore()
metrics = [metric_iou, metric_f1, 'accuracy']
model.compile(
    optimizer=optimizer,
    loss=total_loss,
    metrics=[metrics]
)
# STEP 10: fit the model
history = model.fit(
    train_dataset,
    verbose=2,
    epochs=25,
    validation_data=valid_dataset,
    callbacks=callbacks)

# STEP 11: evaluate the model on the unseen test set
print('Now we can evaluate on the test set:')
model.evaluate(test_dataset)


# STEP 12: visualize the results
def create_label_colormap():
    """Creates a label colormap used in Cityscapes segmentation benchmark.

    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0]], dtype=np.uint8)
    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


def vis_segmentation(image, mask, pred_mask):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(20, 5))
    grid_spec = gridspec.GridSpec(1, 5, width_ratios=[6, 6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    image = tf.keras.preprocessing.image.array_to_img(image)
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    gt_mask = label_to_color_image(mask).astype(np.uint8)
    plt.imshow(gt_mask)
    plt.axis('off')
    plt.title('ground-truth mask')

    plt.subplot(grid_spec[2])
    predicted_mask = label_to_color_image(pred_mask).astype(np.uint8)
    plt.imshow(predicted_mask)
    plt.axis('off')
    plt.title('predicted mask')

    plt.subplot(grid_spec[3])
    plt.imshow(image)
    plt.imshow(predicted_mask, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    ax = plt.subplot(grid_spec[4])
    plt.imshow(FULL_COLOR_MAP.astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(N_CLASSES), LABEL_NAMES)
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()


def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predicitons
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [IMG_HEIGHT, IMG_WIDTH, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        belonging to a certain class. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [IMG_SIZE, IMG_SIZE, 1] segmentation mask with top 1 predictions
        for each pixel.
    """
    # pred_mask -> [IMG_HEIGHT, IMG_WIDTH, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax at encoded axis
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib needs [IMG_SIZE, IMG_SIZE,1 ]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask


def show_predictions(dataset, num):
    """Show num sample predictions.

    Parameters
    ----------
    dataset : [tf.data.dataset]
        e.g. test_dataset
    num : int
        Number of sample to show
    """

    for image, mask in dataset.take(num):
        sample_image, sample_mask = image, mask
        # The model is expecting a tensor of the size
        # [BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3]
        # but sample_image[0] is [IMG_HEIGHT, IMG_WIDTH, 3]
        # and we want only 1 inference to be faster
        # so we add an additional dimension [1, IMG_HEIGHT, IMG_WIDTH, 3]
        # one_img_batch -> [1, IMG_HEIGHT, IMG_WIDTH, 3]
        one_img_batch = sample_image[0][tf.newaxis, ...]
        # inference -> [1, IMG_HEIGHT, IMG_WIDTH, N_CLASS]
        inference = model.predict(one_img_batch)
        # pred_mask -> [1, IMG_HEIGHT, IMG_WIDTH, 1]
        pred_mask = create_mask(inference)
        # decode the one-hot ground-truth mask to get [1, IMG_HEIGHT, IMG_WIDTH, 1]
        mask = create_mask(sample_mask)
        # create slices of the image, mask and pred_mask -> [IMG_HEIGHT, IMG_WIDTH, 1/3]
        vis_segmentation(image[0], mask[0][:, :, 0], pred_mask[0][:, :, 0])


show_predictions(dataset=test_dataset, num=4)

# STEP 13: Plot training & validation iou_score values to visualize training process
plt.figure(figsize=(16, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


def evaluate_dataset(dataset):
    """Evaluate a whole dataset with the MODEL loaded, for some specific labels.

    PARAMETERS:
    dataset: tf.data.dataset

    OUTPUT:
    Returns a table in the console specific class_iou values
    """
    acc_list = []
    intersection_list = []
    union_list = []
    print('evaluating the dataset...')
    for image, mask in dataset:
        sample_image, sample_mask = image, mask
        one_img_batch = sample_image[0][tf.newaxis, ...]
        inference = model.predict(one_img_batch)
        pred_mask = create_mask(inference)
        mask = create_mask(sample_mask)
        seg_map = np.array(pred_mask[0][:, :, 0]).astype(int)
        ground_truth = np.array(mask[0][:, :, 0]).astype(int)
        # merge some labels
        seg_map[np.logical_or(seg_map == 14, seg_map == 15)] = 13
        seg_map[np.logical_or(seg_map == 3, seg_map == 4)] = 2
        seg_map[seg_map == 12] = 11

        # calculate accuracy on valid area
        acc = (np.sum(seg_map[ground_truth != 19] == ground_truth[ground_truth != 19]) / np.sum(ground_truth != 19))
        acc_list.append(acc)
        # select valid labels for evaluation
        cm = confusion_matrix(ground_truth[ground_truth != 19], seg_map[ground_truth != 19],
                              labels=np.array([0, 1, 2, 5, 6, 7, 8, 9, 11, 13, 18]))
        intersection = (np.diag(cm))
        intersection_list.append(intersection)
        union = (np.sum(cm, 0) + np.sum(cm, 1) - np.diag(cm))
        union_list.append(union)
    class_iou = np.round(np.sum(intersection_list, 0) / np.sum(union_list, 0), 4)
    print('pixel accuracy: %.4f' % np.mean(acc_list))
    print('mean class IoU: %.4f' % np.mean(class_iou))
    print('class IoU:')
    print(tabulate([class_iou], headers=LABEL_NAMES[[0, 1, 2, 5, 6, 7, 8, 9, 11, 13, 18]]))


evaluate_dataset(dataset=test_dataset)
# Last step: save the model
# model.save('demo_20classes.hdf5')
