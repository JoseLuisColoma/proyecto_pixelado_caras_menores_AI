import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
import io

app = Flask(__name__)

# Configuración de desenfoque
BLUR_KERNEL_SIZE = (99, 99)
BLUR_SIGMA = 30


@app.route('/pixelate', methods=['POST'])
def pixelate():
    image_bytes = obtener_imagen_bytes(request)
    if image_bytes is None:
        return jsonify({"error": "No se recibió imagen de rostro"}), 400

    image = decodificar_imagen(image_bytes)
    if image is None:
        return jsonify({"error": "No se pudo decodificar la imagen"}), 422

    try:
        pixelada = aplicar_pixelado(image)
        _, buffer = cv2.imencode('.jpg', pixelada)
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/jpeg',
            as_attachment=False,
            download_name='blurred.jpg'
        )
    except Exception as e:
        return jsonify({"error": "Error interno en pixelado", "detalle": str(e)}), 500


# --- Funciones auxiliares ---

def obtener_imagen_bytes(req):
    """Lee los bytes de la imagen del request."""
    if 'face' not in req.files:
        return None
    try:
        return req.files['face'].read()
    except Exception as e:
        print(f"Error al leer imagen: {e}")
        return None

def decodificar_imagen(image_bytes):
    """Convierte los bytes en una imagen OpenCV."""
    try:
        np_img = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error al decodificar imagen: {e}")
        return None

def aplicar_pixelado(imagen):
    """Aplica desenfoque tipo GaussianBlur."""
    return cv2.GaussianBlur(imagen, BLUR_KERNEL_SIZE, BLUR_SIGMA)

# Health check endpoint
# Este endpoint es útil para verificar que el servicio está activo y funcionando.
# Probarlo con: curl http://localhost:5003/health
# Respuesta esperada: {"status": "ok"}
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003)
