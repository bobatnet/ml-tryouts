import pprint
from functools import partial

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import resnet
from tensorflow.python.keras.applications.resnet import ResNet, imagenet_utils

resnet.layers = resnet.VersionAwareLayers()

img_dim = 224


def stacks(x):
    x = resnet.stack1(x, 64, 3, stride1=1, name='conv2')
    x = resnet.stack1(x, 128, 4, name='conv3')
    x = resnet.stack1(x, 256, 6, name='conv4')
    x = resnet.stack1(x, 512, 3, name='conv5')
    return x


input_name = 'my_input'

inp = tf.keras.layers.Input(shape=(img_dim, img_dim, 3), name=input_name)

model = ResNet(stacks, preact=False, use_bias=True,
               model_name='resnet50', weights='imagenet',
               input_tensor=inp, pooling='avg', input_shape=(img_dim, img_dim, 3))

y = model(inp)
print(tf.shape(y))

print(len(model.layers))

names = {}


def show_layers():
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
        names[layer.name] = i

        getcode = lambda x: names.get(x.split('/')[0], -1)
        try:
            if isinstance(layer.input, list):
                for inp in layer.input:
                    print("  ", getcode(inp.name), inp.name)
            else:
                print("  ", getcode(layer.input.name), layer.input.name)
                pass
        except AttributeError:
            pass


# show_layers()
newinp1 = tf.keras.layers.Input(shape=(img_dim, img_dim, 3))
newinp2 = tf.keras.layers.Input(shape=(img_dim, img_dim, 3))

tensors1, tensors2 = {input_name: newinp1}, {input_name: newinp2}

attend_pos = ['conv2_block3_add', 'conv3_block4_add', 'conv4_block6_add', 'conv5_block3_add']
attention_layers = []

layer = None

patch_size = 4

def add_attention(inp_query, inp_key):
    patch_extractor = partial(tf.image.extract_patches, sizes=[1, patch_size, patch_size, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='SAME')

    p1, p2 = patch_extractor(images=inp_query), patch_extractor(images=inp_key)
    mha = tf.keras.layers.MultiHeadAttention(num_heads=32, key_dim=3, value_dim=3, dropout=0.3)
    q = mha(query=p1, key=p2, value=inp_query)
    return q

for i, layer in enumerate(model.layers):
    if i == 0:
        continue
    layer_ins = [layer.input] if not isinstance(layer.input, list) else layer.input
    layer_args_tensors_1, layer_args_tensors_2 = [], []

    for in_ in layer_ins:
        name = in_.name.split('/')[0] if '/' in in_.name else in_.name
        layer_args_tensors_1.append(tensors1[name])
        layer_args_tensors_2.append(tensors2[name])

    if len(layer_args_tensors_1) > 1:
        y1, y2 = layer(layer_args_tensors_1), layer(layer_args_tensors_2)
    else:
        y1, y2 = layer(layer_args_tensors_1[0]), layer(layer_args_tensors_2[0])

    tensors1[layer.name], tensors2[layer.name] = y1, y2

    if layer.name in attend_pos:
        print(layer.name, y1.shape)
        y1a = add_attention(inp_query=y1, inp_key=y2)
        y2a = add_attention(inp_query=y2, inp_key=y1)

        tensors1[layer.name], tensors2[layer.name] = y1a, y2a

out_ = tf.keras.layers.Multiply()([tensors1[layer.name], tensors2[layer.name]])
out = tf.keras.layers.Dense(2)(out_)

print("Output", out.shape)
