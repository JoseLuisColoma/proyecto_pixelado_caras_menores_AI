import numpy as np
import cv2
from flask import Flask, request, jsonify
from keras.models import load_model

app = Flask(__name__)

# --- Configuración global ---
MODEL_PATH = "modelo/modelo_mnet.keras"
IMG_SIZE = (200, 200)
PREDICTION_THRESHOLD = 0.38

# --- Cargar modelo ---
model = load_model(MODEL_PATH)

@app.route('/classify', methods=['POST'])
def classify():
    image_bytes = obtener_imagen_bytes(request)
    if image_bytes is None:
        return jsonify({"error": "No se recibió una imagen de rostro"}), 400

    image = decodificar_imagen(image_bytes)
    if image is None:
        return jsonify({"error": "La imagen no pudo ser decodificada"}), 422

    try:
        procesada = preprocesar_imagen(image, IMG_SIZE)
        probabilidad = float(model.predict(procesada, verbose=0)[0][0])
        es_menor = probabilidad > PREDICTION_THRESHOLD

        return jsonify({
            "is_minor": es_menor,
            "score": round(probabilidad, 4)
        })

    except Exception as e:
        return jsonify({"error": "Error al procesar el rostro", "detalle": str(e)}), 500

# --- Funciones auxiliares ---

def obtener_imagen_bytes(req):
    """Extrae y lee la imagen desde el request."""
    if 'face' not in req.files:
        return None
    try:
        return req.files['face'].read()
    except Exception as e:
        print(f"Error al leer archivo: {e}")
        return None

def decodificar_imagen(image_bytes):
    """Convierte los bytes en una imagen OpenCV."""
    try:
        np_img = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error al decodificar imagen: {e}")
        return None

def preprocesar_imagen(imagen, size):
    """Redimensiona y normaliza la imagen para el modelo."""
    imagen = cv2.resize(imagen, size)
    imagen = imagen.astype("float32") / 255.0
    return np.expand_dims(imagen, axis=0)

# Health check endpoint
# Este endpoint es útil para verificar que el servicio está activo y funcionando.
# Probarlo con: curl http://localhost:5002/health
# Respuesta esperada: {"status": "ok"}
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

# --- Ejecutar servidor ---
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002)
