from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import base64
from tensorflow.keras.layers import DepthwiseConv2D as _DepthwiseConv2D

# تعريف طبقة DepthwiseConv2D مخصصة لتجاهل المفتاح 'groups'
class DepthwiseConv2D(_DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

app = Flask(__name__)

# تحميل النموذج مع تجاوز مشكلة groups
model = tf.keras.models.load_model(
    'model.h5',
    custom_objects={'DepthwiseConv2D': DepthwiseConv2D}
)

# دالة لفك Base64 إلى صورة OpenCV
def decode_image(img_data_b64):
    header, encoded = img_data_b64.split(',', 1)
    data = base64.b64decode(encoded)
    np_arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# دالة تجهيز الصورة قبل التنبؤ
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img = decode_image(data['image'])
    x = preprocess(img)
    preds = model.predict(x)[0]
    letter = 'H' if np.argmax(preds) == 0 else 'I'
    return jsonify({'letter': letter})

@app.route('/blind')
def blind():
    return render_template('blind.html')

if __name__ == '__main__':
    app.run(debug=True)
