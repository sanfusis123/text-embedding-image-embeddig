import tensorflow as tf
from sklearn.decomposition import PCA

pca = PCA(n_components=1024)

base_model = tf.keras.applications.ResNet152V2(
    include_top=True,
    weights='imagenet',
    input_shape = (224,224,3),
    classifier_activation='linear'
)

def get_model(base_model):
    input_x = tf.keras.Input(shape = (224,224,3))
    x = tf.keras.Model(inputs = base_model.inputs, outputs = base_model.get_layer('conv4_block10_out').output)(input_x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs = input_x, outputs = x)

# model = get_model(base_model)
model = base_model
print(model.summary())
def load_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, (224,224))
    return image/255.


def get_image_embeddings(img_path):
    image = load_image(img_path)
    pred = model.predict(tf.expand_dims(image, axis=0))
    pred = tf.squeeze(pred)
    return pred
