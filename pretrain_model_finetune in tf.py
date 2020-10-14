import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import activations, optimizers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import random
from glob import glob
import argparse
import matplotlib.pyplot as plt


def args():
    parser = argparse.ArgumentParser(description='project training configurations')

    parser.add_argument('--net_type', type=str, default='resnet50')
    parser.add_argument('--train_data_path', type=str, default='../dataset/train_data', help='training file name')
    parser.add_argument('--target_size', type=int, default=224, help='target_width and target_height of input image')
    parser.add_argument('--batch_size', type=int, default=64, help='traing batch size')
    parser.add_argument('--validation_split', type=float, default=0.1, help='train_test split ratio')
    parser.add_argument('--epochs', type=int, default=0.1, help='training epochs')
    # parser.add_argument('--')

    return parser.parse_args()


# pretrain model already downloaded from tensorflow_hub in model directory
def get_pre_train_model(net_type):
    net_type_list = ['resnet50','resnet101', 'efficientnetb0', 'efficientnetb2', 'efficientnetb4', 'efficientnetb7']

    if net_type not in net_type_list:
        raise Exception('the model is out of current scope')

    model_dir = 'models/'
    model_path = model_dir + net_type
    model = keras.models.load_model(model_path)
    return model


def make_image_generator():
    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=15,
                                       width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, vertical_flip=True,
                                       fill_mode='nearest')

    # test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    return train_datagen


def make_train_dataset(generator, data_dir, batch_size, image_size):
    dataset = generator.flow_from_directory(data_dir,
                                            target_size=image_size,
                                            color_mode='rgb',
                                            shuffle=True,
                                            batch_size=batch_size)
    return dataset


class FineTuneModel(keras.models.Model):
    def __init__(self, pre_train_model, num_class, **kwargs):
        super().__init__(**kwargs)
        self.pre_train_model = pre_train_model
        self.dense = layers.Dense(units=num_class, activations='softmax')

    def call(self, inputs):
        x = self.pre_train_model(inputs)
        outputs = self.dense(x)
        return outputs


def main():
    arguments = args()
    net_type = arguments.net_type
    train_data_path = arguments.train_data_path
    target_size = arguments.target_size
    batch_size = arguments.batch_size
    epochs = arguments.epochs
    validation_split = arguments.validation_split
    image_size = (target_size, target_size)

    random.seed(32)
    image_fns = glob(os.path.join(train_data_path, '*', '*.*'))
    random.shuffle(image_fns)
    # train_fns, val_fns = train_test_split(image_fns, test_size=validation_split, shuffle=True)

    generator = make_image_generator()
    data_generator = make_train_dataset(generator, train_data_path, batch_size, target_size)

    checkpoint_path = 'checkpoints/cp.ckpt'
    checkpoint_dir = os.path.basename(checkpoint_path)
    checkpoint_cb = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    monitor='val_accuracy',
                                                    save_best_only=True)
    checkpoint_lr_cb = keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                         monitor='val_loss',
                                                         patience=10)

    label_names = [s.split('/')[-2] for s in image_fns]
    unique_labels = list(set(label_names))
    num_classes = len(unique_labels)
    # unique_labels.sort()
    # id_labels = {_id: name for name, _id in enumerate(unique_labels)}

    ###################################################################################

    pre_train_model = get_pre_train_model(net_type)

    #  构建模型的其他两种方法， 这两种方法的好处是可以freeze pretrain model weight 进行计算
    #  注意freeze weights 要在model.fit 之前
    # pretrain_model_feat_dim  = pre_train_model.output.shape[0]
    # model = tf.keras.sequential(
    #     pre_train_model,
    #     keras.layer.Dense(units=len(unique_labels), activations='softmax')
    # )

    # model = keras.models.Sequential([])
    # model.add(pretrain_model)
    # model.add( keras.layers.Dense(num_classes, activation='softmax')

    model = FineTuneModel(pre_train_model, num_class=num_classes)

    if tf.train.latest_checkpoint(checkpoint_dir):
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['acc', 'val_acc'])

    history = model.fit(data_generator,  # steps_per_epoch=100,  # 2000 images = batch_size * steps,s
                        epochs=20,  # validation_steps=50,  # 1000 images = batch_size * steps
                        validation_split=validation_split,
                        callbacks=[checkpoint_cb, checkpoint_lr_cb],
                        verbose=2)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 12))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')


if __name__ == '__main__':
    main()


# code sample：数据增强的另一种写法
# ds = tf.data.Dataset.from_generator(
#     img_gen.flow_from_directory, args=[flowers],
#     output_types=(tf.float32, tf.float32),
#     output_shapes=([32,256,256,3], [32,5])
# )

# data_augmentation = tf.keras.Sequential([
#   tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
#   tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
#   tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
# ])
# resize_and_rescale = tf.keras.Sequential([
#   layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
#   layers.experimental.preprocessing.Rescaling(1./255)
# ])

