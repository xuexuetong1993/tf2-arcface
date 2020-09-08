import tensorflow as tf


def _ImageNetCrop(image, input_size):
    """Imagenet-style crop with random bbox and aspect ratio.
    Args:
      image: a `Tensor`, image to crop.
    Returns:
      cropped_image: `Tensor`, cropped image.
    """

    min_object_covered = 0.1
    aspect_ratio_range_min = 3. / 4
    aspect_ratio_range_max = 4. / 3
    area_range_min = 0.08
    area_range_max = 1.0
    max_attempts = 100
    update_labels = False
    # 'central_fraction' is used for central crop in inference.
    central_fraction = 0.875
    random_reflection = False
    input_rows = input_size
    input_cols = input_size

    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    (bbox_begin, bbox_size, _) = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=(aspect_ratio_range_min,
                            aspect_ratio_range_max),
        area_range=(area_range_min, area_range_max),
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    cropped_image.set_shape([None, None, 3])

    cropped_image = tf.image.resize(
        cropped_image, [input_rows, input_cols], method='area')
    if random_reflection:
        cropped_image = tf.image.random_flip_left_right(cropped_image)
    print("Using distorted crop func")
    return cropped_image

def _parse_tfrecord(binary_img=False, is_ccrop=False, input_size=112):
    def parse_tfrecord(tfrecord):
        if binary_img:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/filename': tf.io.FixedLenFeature([], tf.string),
                        'image/encoded': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        else:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/img_path': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            image_encoded = tf.io.read_file(x['image/img_path'])
            x_train = tf.image.decode_jpeg(image_encoded, channels=3)

        y_train = tf.cast(x['image/source_id'], tf.float32)

        x_train = _transform_images(is_ccrop=is_ccrop, input_size=input_size)(x_train)
        y_train = _transform_targets(y_train)
        return (x_train, y_train), y_train
    return parse_tfrecord


def _transform_images(is_ccrop=False, input_size=112):
    def transform_images(x_train):
        #x_train = tf.image.resize(x_train, (input_size, input_size))
        x_train = _ImageNetCrop(x_train, input_size)
        #x_train = tf.image.random_crop(x_train, (112, 112, 3))
        x_train = tf.image.random_flip_left_right(x_train)
        #x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
        #x_train = tf.image.random_brightness(x_train, 0.4)
        #x_train = x_train / 255
        return x_train
    return transform_images


def _transform_targets(y_train):
    return y_train


def load_tfrecord_dataset(tfrecord_name, batch_size,input_size,
                          binary_img=False, shuffle=True, buffer_size=10240,
                          is_ccrop=False):
    """load dataset from tfrecord"""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfrecord(binary_img=binary_img, is_ccrop=is_ccrop, input_size=input_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def load_fake_dataset(size):
    """load fake dataset"""
    x_train = tf.image.decode_jpeg(
        open('./data/BruceLee.JPG', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)
    x_train = tf.image.resize(x_train, (size, size))

    labels = [0]
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))
