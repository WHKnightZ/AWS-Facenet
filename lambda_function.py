from __future__ import print_function
import os
import boto3
import hmac
import hashlib
import base64
import json
import numpy as np
import cv2
from scipy.spatial import distance

ENDPOINT_NAME = os.environ['ENDPOINT_NAME']

client = boto3.client('runtime.sagemaker')


def detect_faces(image, margin):
    '''
    sử dụng detectMultiScale của opencv để detect ra tọa độ các khuôn mặt từ ảnh image
    :return: danh sách các khuôn mặt
    '''
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    #     faces = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4)

    aligned_images = []
    margin_half = margin // 2

    for face in faces:
        (x, y, w, h) = face
        cropped = image[y - margin_half:y + h + margin_half, x - margin_half:x + w + margin_half, :]
        aligned = cv2.resize(cropped, (160, 160))
        aligned_images.append(aligned)

    return faces, np.array(aligned_images)


def prewhiten(x):
    '''
    tiền xử lý khuôn mặt trước khi đưa vào mô hình
    '''
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def l2_normalize(x, axis=-1, epsilon=1e-10):
    '''
    chuẩn hóa vector đầu ra
    '''
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def calc_embs(image, margin=10):
    '''
    xử lý ảnh, đưa ảnh về vector
    '''
    faces, aligned_images = detect_faces(image, margin)
    if len(faces) == 0:
        return None, None

    prewhiten_images = prewhiten(aligned_images)

    # invoke endpoint, truyền vào image, endpoint trả về vector đầu ra bức ảnh
    response = client.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='application/json',
                                      Body=json.dumps(prewhiten_images.tolist()))
    predicts = json.loads(response['Body'].read())['predictions']

    embs = []
    for predict in predicts:
        embs.append(l2_normalize(predict).tolist())

    return faces, embs


def cal_distance(vector1, vector2):
    '''
    tính khoảng cách giữa 2 vector
    '''
    return distance.euclidean(vector1, vector2)


def paint(image, box, label):
    '''
    vẽ khung và thêm nhãn cho bức ảnh theo box và label
    '''
    color_box = [0, 192, 0]  # màu xanh làm màu khung
    font = cv2.FONT_HERSHEY_PLAIN
    x, y, w, h = box
    text_width = cv2.getTextSize(label, font, 1.2, 2)[0][0]
    cv2.rectangle(image, (x, y), (x + w, y + h), color_box, 2)
    cv2.putText(image, label, (x + 10, y + 25), font, 1.2, (255, 255, 255), 1, cv2.LINE_AA)


def predict(image):
    '''
    dự đoán tên người trong bức ảnh được truyền vào
    '''
    # trước tiên mở file data.json để lấy dữ liệu đã train: các khuôn mặt kèm theo các vector
    with open('data.json') as json_file:
        data = json.load(json_file)['data']

    # lấy ra tất cả các box của ảnh
    boxes, embs = calc_embs(image)

    if boxes is None:
        return None

    result = []
    for box, emb in zip(boxes, embs):
        min_value = 99999
        label = None

        # quét toàn bộ vector đã lấy từ file json, xem vector nào gần với khuôn mặt này nhất
        for vectors in data:  # vectors là danh sách các tên người kèm các vector
            for vector in vectors['values']:  # mỗi tên lại có nhiều vector (khuôn mặt)
                dis = cal_distance(vector, emb)
                if dis < min_value:
                    min_value = dis
                    label = vectors['name']
        paint(image, box, label)  # tìm được box và label rồi thì vẽ vào ảnh
        result.append({'box': box.tolist(), 'label': label})

    return result


def send_results(json_body,
                 status_code=200,
                 headers={
                     'Content-Type': 'application/json',
                     "Access-Control-Allow-Origin": "*",
                     "Access-Control-Allow-Credentials": True
                 }):
    return {
        'statusCode': status_code,
        'headers': headers,
        'body': json.dumps(json_body)
    }


def lambda_handler(event, context):
    request_data = json.loads(event['body'])
    keys = request_data.keys()

    if 'image' not in keys:
        return send_results({'msg': 'Please input image (base64)'})

    data = request_data['image']
    encoded_data = data.split(',')[1]  # cắt lấy phần nội dung của ảnh

    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)  # chuyển base64 => mảng
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # đọc ảnh từ mảng

    value = predict(image)

    if not value:
        return send_results({'msg': 'Image does not contain any faces'})

    # đưa ảnh đã vẽ thêm box, label về base64 để trả về
    image_base64 = 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()

    return send_results({'image': image_base64, 'predicts': value})
