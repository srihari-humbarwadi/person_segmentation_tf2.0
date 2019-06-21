import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from deeplab_test import DeepLabV3Plus
from tqdm import tqdm_notebook

print('TensorFlow', tf.__version__)


H, W = 1280, 1280

train_images = sorted(glob('resized_images/*'))
train_masks = sorted(glob('resized_masks/*'))

val_images = sorted(glob('validation_data/images/*'))
val_masks = sorted(glob('validation_data/masks/*'))

print(f'Found {len(train_images)} training images')
print(f'Found {len(train_masks)} training masks')

print(f'Found {len(val_images)} validation images')
print(f'Found {len(val_masks)} validation masks')

for i in range(len(train_masks)):
    assert train_images[i].split(
        '/')[-1].split('.')[0] == train_masks[i].split('/')[-1].split('.')[0]

for i in range(len(val_masks)):
    assert val_images[i].split(
        '/')[-1].split('.')[0] == val_masks[i].split('/')[-1].split('.')[0]


model = DeepLabV3Plus(H, W)
model.load_weights('top_weights.h5')


def pad_inputs(image, crop_height=H, crop_width=H, pad_value=0):
    dims = tf.cast(tf.shape(image), dtype=tf.float32)
    h_pad = tf.maximum(crop_height - dims[0], 0)
    w_pad = tf.maximum(crop_width - dims[1], 0)
    padded_image = tf.pad(image, paddings=[[0, h_pad], [0, w_pad], [
                          0, 0]], constant_values=pad_value)
    return padded_image, h_pad, w_pad


def resize_preserve_aspect_ratio(image_tensor, max_side_dim):
    img_h, img_w = image_tensor.shape.as_list()[:2]
    min_dim = tf.maximum(img_h, img_w)
    resize_ratio = max_side_dim / min_dim
    new_h, new_w = resize_ratio * img_h, resize_ratio * img_w
    resized_image_tensor = tf.image.resize(
        image_tensor, size=[new_h, new_w], method='bilinear')
    return resized_image_tensor


def prepare_inputs(image_path, H=H, W=W, maintain_resolution=False):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape([None, None, 3])
    shape = image.shape.as_list()[:2]
    if maintain_resolution:
        disp_image = image.numpy().copy()
    image = tf.cast(image, dtype=tf.float32)
    resized = False
    if tf.maximum(shape[0], shape[1]) > H:
        resized = True
        image = resize_preserve_aspect_ratio(image, max_side_dim=H)
    image, h_pad, w_pad = pad_inputs(image)
    if not maintain_resolution:
        disp_image = image.numpy().copy()
    image = image[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    return disp_image, tf.cast(image, dtype=tf.float32), np.int32(h_pad.numpy()), np.int32(w_pad.numpy()), resized


def resize_mask(mask, size):
    mask = tf.image.resize(mask[..., None], size, method='nearest')
    return mask[..., 0]


def pipeline(image_path, alpha=0.7, maintain_resolution=False):
    disp_image, image, h_pad, w_pad, resized = prepare_inputs(
        image_path, maintain_resolution=maintain_resolution)
    mask = model(image[None, ...])[0, ..., 0] > 0.5
    mask = tf.cast(mask, dtype=tf.uint8)
    b_h, b_w = (image.shape[:2] - tf.constant([h_pad, w_pad])).numpy()
    disp_mask = mask[:b_h, :b_w].numpy()
    if resized and maintain_resolution:
        disp_mask = resize_mask(disp_mask, disp_image.shape[:2]).numpy()
    else:
        disp_image = disp_image[:b_h, :b_w]
    overlay = disp_image.copy()
    overlay[disp_mask == 0] = [255, 0, 0]
    overlay[disp_mask == 1] = [0, 0, 255]
    cv2.addWeighted(disp_image, alpha, overlay, 1 - alpha, 0, overlay)
    extracted_pixels = disp_image.copy()
    extracted_pixels[disp_mask == 0] = [207, 207, 207]
    return np.uint8(disp_image), np.uint8(np.concatenate([disp_mask[..., None]] * 3, axis=-1) * 255), np.uint8(overlay), np.uint8(extracted_pixels)


test_images = glob('validation_data/images/*')


for img in tqdm_notebook(test_images[10:]):
    output = pipeline(img, maintain_resolution=False)
    result = np.concatenate(output, axis=1)
    fname = img.split('/')[-1].split('.')[0] + '.png'
    cv2.imwrite(f'_1024/{fname}', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
