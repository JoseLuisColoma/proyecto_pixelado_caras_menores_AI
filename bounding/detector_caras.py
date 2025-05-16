import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Cargar el modelo de detección de caras (HaarCascade)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)


@app.route('/detect', methods=['POST'])
def detect_faces():
    image_bytes = obtener_imagen_bytes(request)
    if image_bytes is None:
        return jsonify({"error": "No se recibió ninguna imagen."}), 400

    image = decodificar_imagen(image_bytes)
    if image is None:
        return jsonify({"error": "No se pudo decodificar la imagen."}), 422

    try:
        faces = detectar_caras(image)
        return jsonify({"faces": faces})
    except Exception as e:
        return jsonify({"error": "Error interno al detectar rostros.", "detalle": str(e)}), 500


def obtener_imagen_bytes(req):
    """Extrae y lee los bytes de la imagen del request."""
    if 'image' not in req.files:
        return None
    try:
        return req.files['image'].read()
    except Exception as e:
        print(f"Error al leer imagen: {e}")
        return None


def decodificar_imagen(image_bytes):
    """Convierte los bytes en una imagen OpenCV."""
    try:
        np_img = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error al decodificar la imagen: {e}")
        return None


def detectar_caras(img):
    """Detecta las caras en una imagen usando HaarCascade"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detecciones = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )
    return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in detecciones]


# Health check endpoint
# Este endpoint es útil para verificar que el servicio está activo y funcionando.
# Probarlo con: curl http://localhost:5001/health
# Respuesta esperada: {"status": "ok"}
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)