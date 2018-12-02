import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History, Callback
from tensorflow.contrib.tpu.python.tpu import keras_support
import tensorflow.keras.backend as K

from ricap import ricap
import os, pickle

# RICAP Generator
class RICAPGenerator(ImageDataGenerator):
    def __init__(self, ricap_beta=0.3, use_batchwise_random=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ricap_beta = ricap_beta
        self.use_batchwise_random = use_batchwise_random

    def flow_from_directory(self, *args, **kwargs):
        for batch_X, batch_y in super().flow_from_directory(*args, **kwargs):
            ricap_X, ricap_y = ricap(batch_X, batch_y, self.ricap_beta, self.use_batchwise_random)
            yield ricap_X, ricap_y

#create transfer cnn
def create_network():
    net = DenseNet121(include_top=False, weights="imagenet", input_shape=(160,160,3))

    # don't train until conv4 blocks
    for l in net.layers:
        if "conv4" in l.name: break
        l.trainable = False

    x = GlobalAveragePooling2D()(net.layers[-1].output)
    x = Dense(176, activation="softmax")(x)

    return Model(net.inputs, x)

class RICAPBetaCallback(Callback):
    def __init__(self, generator):
        self.gen = generator

    def on_epoch_end(self, epoch, logs):
        if epoch < 20:
            self.gen.ricap_beta = 0.01
        elif epoch < 90:
            self.gen.ricap_beta = (epoch-20.0) / (90-20.0) * 0.99 + 0.01
        else:
            self.gen.ricap_beta = 1.0

def train(use_batchwise_random):
    batch_size=1024
    train_gen_instance = RICAPGenerator(rescale=1.0/255, width_shift_range=15.0/160, 
                                        height_shift_range=15.0/160, horizontal_flip=True, ricap_beta=0.01,
                                        use_batchwise_random=use_batchwise_random)
    train_gen = train_gen_instance.flow_from_directory(
        "animeface-character-dataset/train", target_size=(160,160), batch_size=batch_size)
    test_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
        "animeface-character-dataset/test", target_size=(160,160), batch_size=batch_size)

    model = create_network()
    model.compile(tf.train.RMSPropOptimizer(1e-4), "categorical_crossentropy", ["acc"])

    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    hist = History()
    ricap_scheduler = RICAPBetaCallback(train_gen_instance)
    model.fit_generator(train_gen, steps_per_epoch=10062//batch_size, callbacks=[hist, ricap_scheduler],
                        validation_data=test_gen, validation_steps=4428//batch_size, epochs=100)

    history = hist.history
    with open(f"anime_ricap_batchwise_{use_batchwise_random}.dat", "wb") as fp:
        pickle.dump(history, fp)

if __name__ == "__main__":
    K.clear_session()
    train(False)
