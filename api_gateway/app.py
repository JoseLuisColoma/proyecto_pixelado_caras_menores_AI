import os
import requests
import io
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Límite de 10MB por imagen

ENGINE_URL = "http://engine:5000/process"

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No se proporcionó ninguna imagen."}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({"error": "Nombre de archivo vacío."}), 400

    if not allowed_file(image.filename):
        return jsonify({"error": "Formato de imagen no permitido. Solo JPG, JPEG o PNG."}), 415

    try:
        image_bytes = image.read()
    except Exception as e:
        return jsonify({"error": "No se pudo leer el archivo.", "detalle": str(e)}), 422

    try:
        response = requests.post(ENGINE_URL, files={'image': image_bytes}, timeout=30)
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "No se pudo conectar al motor (engine)."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Tiempo de espera agotado al contactar con engine."}), 504
    except Exception as e:
        return jsonify({"error": "Error inesperado al contactar con engine.", "detalle": str(e)}), 500

    if response.status_code != 200:
        return jsonify({
            "error": "El motor (engine) devolvió un error.",
            "detalle": response.text
        }), response.status_code

    try:
        return send_file(
            io.BytesIO(response.content),
            mimetype='image/jpeg',
            as_attachment=False,
            download_name='procesada.jpg'
        )
    except Exception as e:
        return jsonify({"error": "Error al devolver la imagen procesada.", "detalle": str(e)}), 500

# Health check endpoint
# Este endpoint es útil para verificar que el servicio está activo y funcionando.
# Probarlo con: curl http://localhost:5000/health
# Respuesta esperada: {"status": "ok"}
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"error": "Archivo demasiado grande. Máximo 10MB."}), 413

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)