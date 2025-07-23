from flask import Flask, render_template, request, jsonify
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Path ke file model .h5
MODEL_PATH = 'model/model_jambu_biji.h5'

# Konfigurasi Model
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CLASS_NAMES = ['mentah', 'setengah matang', 'matang', 'busuk', 'bukan jambu biji merah']

# Deskripsi tambahan untuk setiap kelas
CLASS_DESCRIPTIONS = {
    'mentah': "Buah masih mentah dengan warna hijau pekat dan tesktur keras",
    'setengah matang': "Buah setengah matang dengan warna hijau kekuningan dan tekstur lunak di beberapa bagian",
    'matang': "Buah sudah matang dengan warna kuning dan tekstur lunak",
    'busuk': "Buah sudah busuk dengan warna terlihat hitam dan kecoklatan dengan tekstur lembek cenderung hancur",
    'bukan jambu biji merah': "Buah tidak dikenali sebagai buah jambu biji merah"
}

# Muat model saat aplikasi dimulai
model = None
try:
    model = load_model(MODEL_PATH)
    print(f"Model berhasil dimuat dari {MODEL_PATH}")
except Exception as e:
    print(f"Gagal memuat model: {e}")

# Fungsi untuk preprocessing gambar
def preprocess_image(img):
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = np.array(img)

    # Pastikan gambar tidak memiliki alpha channel
    if img_array.shape[-1] == 4:  # Jika PNG dengan alpha channel
        img_array = img_array[:, :, :3]

    # Normalisasi
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/deteksi', methods=['GET', 'POST'])
def deteksi():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file yang diunggah'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

        if file and allowed_file(file.filename):
            try:
                img = Image.open(file.stream).convert('RGB')
                img_array = preprocess_image(img)

                if model is not None:
                    prediction = model.predict(img_array, verbose=0)

                    if prediction.shape[1] != len(CLASS_NAMES):
                        return jsonify({'error': 'Jumlah kelas model tidak sesuai'}), 500

                    predicted_class = np.argmax(prediction)
                    confidence = float(prediction[0][predicted_class])
                    hasil_deteksi = CLASS_NAMES[predicted_class]
                    deskripsi = CLASS_DESCRIPTIONS[hasil_deteksi]

                    confidence_threshold = 0.7
                    if confidence < confidence_threshold:
                        hasil_deteksi = 'bukan jambu biji merah'
                        deskripsi = CLASS_DESCRIPTIONS[hasil_deteksi]

                    return jsonify({
                        'hasil': hasil_deteksi,
                        'deskripsi': deskripsi,
                        'confidence': confidence
                    })
                else:
                    return jsonify({'error': 'Model tidak tersedia'}), 503

            except Exception as e:
                return jsonify({'error': f'Terjadi kesalahan saat memproses gambar: {e}'}), 500

        return jsonify({'error': 'Format file tidak diizinkan'}), 400

    return render_template('deteksi.html')

@app.route('/tentang')
def tentang():
    return render_template('tentang.html')

if __name__ == '__main__':
    app.run(debug=True)
