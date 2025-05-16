#el contenedor engine actúa como un orquestador. Este servicio:
#Recibe la imagen desde el api_gateway.
#La reenvía al bounding para detectar rostros.
#Por cada rostro detectado, lo envía al classifier.
#Si es menor, lo envía al pixelator para modificar la imagen.

# engine/engine.py

import cv2
import os
import numpy as np
import requests
import io
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)

# URL del servicio de detección de rostros (bounding)
BOUNDING_URL = "http://bounding:5001/detect"
# URL de los servicios de clasificación y pixelado
CLASSIFIER_URL = "http://classifier:5002/classify"
# URL del servicio de pixelado
PIXELATOR_URL = "http://pixelator:5003/pixelate"


@app.route('/process', methods=['POST'])
def process():
    image_bytes = obtener_imagen_bytes(request)
    if image_bytes is None:
        return jsonify({"error": "No se recibió ninguna imagen."}), 400

    image = decodificar_imagen(image_bytes)
    if image is None:
        return jsonify({"error": "No se pudo decodificar la imagen."}), 422

    faces = detectar_rostros(image_bytes)
    if faces is None:
        return jsonify({"error": "Error en el servicio de detección de caras."}), 503

    for face in faces:
        image = procesar_rostro(image, face)

    return generar_respuesta_imagen(image)


def obtener_imagen_bytes(req):
    if 'image' not in req.files:
        return None
    try:
        return req.files['image'].read()
    except Exception as e:
        print(f"Error al leer la imagen: {e}")
        return None

def decodificar_imagen(image_bytes):
    try:
        npimg = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error al decodificar la imagen: {e}")
        return None

def detectar_rostros(image_bytes):
    try:
        resp = requests.post(BOUNDING_URL, files={'image': image_bytes}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("faces", [])
    except Exception as e:
        print(f"Error al contactar con el bounding: {e}")
        return None

def procesar_rostro(imagen, face):
    try:
        x, y, w, h = face['x'], face['y'], face['w'], face['h']
        rostro = imagen[y:y+h, x:x+w]
        _, buffer = cv2.imencode('.jpg', rostro)
        rostro_bytes = buffer.tobytes()

        # Clasificar
        resp_clf = requests.post(CLASSIFIER_URL, files={'face': rostro_bytes}, timeout=10)
        resp_clf.raise_for_status()
        is_minor = resp_clf.json().get("is_minor", False)

        # Si es menor, pixelar
        if is_minor:
            resp_pix = requests.post(PIXELATOR_URL, files={'face': rostro_bytes}, timeout=10)
            resp_pix.raise_for_status()
            rostro_pixelado = cv2.imdecode(np.frombuffer(resp_pix.content, np.uint8), cv2.IMREAD_COLOR)
            if rostro_pixelado is not None:
                imagen[y:y+h, x:x+w] = rostro_pixelado

    except Exception as e:
        print(f"Error procesando la cara: {e}")
    return imagen

def generar_respuesta_imagen(image):
    try:
        _, buffer = cv2.imencode('.jpg', image)
        return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": "Error al generar imagen de salida", "detalle": str(e)}), 500


# Health check endpoint
# Este endpoint es útil para verificar que el servicio está activo y funcionando.
# Probarlo con: curl http://localhost:5000/health
# Respuesta esperada: {"status": "ok"}
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
