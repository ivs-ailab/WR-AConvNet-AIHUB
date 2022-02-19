from absl import logging
from absl import flags
from absl import app

import onnxruntime

import numpy as np
import cv2

import time
import os

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

flags.DEFINE_string('model_path', 'assets/model/weather-recognition.onnx', help='')
flags.DEFINE_string('image_path', 'samples/dust.jpg', help='')

FLAGS = flags.FLAGS


def load_labels(path):
    with open(path, mode='r', encoding='utf-8') as f:
        labels = f.read().split('\n')
    return labels


def imread(path):
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x


def predict(image, session):
    image = cv2.resize(image, dsize=(88, 88), interpolation=cv2.INTER_CUBIC)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    image = np.transpose(image, axes=(2, 0, 1))
    image = np.expand_dims(image, axis=0)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: image.astype(np.float32)})[0]
    output = softmax(output)

    pred = np.argmax(output, 1)

    return pred


def run(model_path, image_path):
    session = onnxruntime.InferenceSession(os.path.join(project_root, model_path))
    class_name = load_labels(os.path.join(os.path.dirname(os.path.join(project_root, model_path)), 'labels.txt'))
    image_path = os.path.join(project_root, image_path)

    image = imread(image_path)
    st = time.time()
    p = predict(image, session)
    et = time.time()
    p = p[0]
    logging.info(f'Prediction: {class_name[p]}')
    logging.info(f'Elapsed Time: {et - st}')
    logging.info(f'FPS: {1 / (et - st)}')


def main(_):
    logging.info('START')

    run(FLAGS.model_path, FLAGS.image_path)

    logging.info('FINISH')


if __name__ == '__main__':
    app.run(main)
