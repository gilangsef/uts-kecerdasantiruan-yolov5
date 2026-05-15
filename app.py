import os
import torch
import pandas as pd
import base64
import cv2
import numpy as np
from io import BytesIO
from flask import Flask, redirect, render_template, request, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Konfigurasi Folder
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png']
app.config['UPLOAD_PATH'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load Model YOLOv5 (Pastikan arahnya ke file model kamu)
model_path = 'runs/train/exp3/weights/best.pt'
model = torch.hub.load('.', 'custom', path=model_path, source='local')

# --- TAMBAHKAN 2 BARIS INI ---
model.conf = 0.05  # Turunkan batas minimal keyakinan AI menjadi 5% saja
model.iou = 0.45   # Standar tumpang tindih deteksi

# Dictionary Informasi Penyakit (Berdasarkan class dataset Roboflow)
DISEASE_INFO = {
    'bacterial_leaf_blight': {
        'nama': 'Hawar Daun Bakteri (Bacterial Leaf Blight)', 
        'status': 'Bahaya', 'color': 'red', 
        'saran': 'Gunakan bakterisida yang tepat, kurangi pupuk Nitrogen, dan perbaiki sistem drainase sawah.'
    },
    'brown_spot': {
        'nama': 'Bercak Coklat (Brown Spot)', 
        'status': 'Peringatan', 'color': 'orange', 
        'saran': 'Perbaiki nutrisi tanah, pastikan tanaman mendapat cukup kalium dan air.'
    },
    'leaf_blast': {
        'nama': 'Blas Daun (Leaf Blast)', 
        'status': 'Bahaya', 'color': 'red', 
        'saran': 'Gunakan fungisida berbahan aktif trisiklazol dan hindari jarak tanam yang terlalu rapat.'
    },
    'leaf_scald': {
        'nama': 'Penyakit Gosong Daun (Leaf Scald)', 
        'status': 'Peringatan', 'color': 'yellow', 
        'saran': 'Gunakan benih yang sehat dan hindari genangan air berlebih di area sawah.'
    },
    'narrow_brown': {
        'nama': 'Bercak Coklat Sempit (Narrow Brown Spot)', 
        'status': 'Waspada', 'color': 'amber', 
        'saran': 'Gunakan varietas padi yang tahan penyakit dan atur kalibrasi pemupukan berimbang.'
    },
    'healthy': {
        'nama': 'Padi Sehat (Healthy)', 
        'status': 'Aman', 'color': 'green', 
        'saran': 'Daun padi dalam kondisi sehat. Pertahankan perawatan dan jadwal pemupukan yang ada!'
    }
}

# def process_detection(image_path):
#     """Fungsi pembantu untuk menjalankan deteksi dan memproses hasil"""
#     results = model(image_path)
#     df = results.pandas().xyxy[0]
    
#     detections = []
#     for index, row in df.iterrows():
#         class_name = row['name']
#         confidence = round(row['confidence'] * 100, 1)
#         info = DISEASE_INFO.get(class_name, {
#             'nama': class_name, 'status': 'Unknown', 'color': 'gray', 'saran': 'Hubungi penyuluh.'
#         })
#         detections.append({
#             'nama': info['nama'],
#             'confidence': confidence,
#             'status': info['status'],
#             'color': info['color'],
#             'saran': info['saran']
#         })
    
#     # Simpan Gambar Hasil
#     results.render()
#     filename = os.path.basename(image_path)
#     result_filename = 'res_' + filename
#     output_path = os.path.join(RESULT_FOLDER, result_filename)
#     Image.fromarray(results.ims[0]).save(output_path)
    
#     return sorted(detections, key=lambda x: x['confidence'], reverse=True), result_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-page')
def upload_page():
    return render_template('upload.html')

@app.route('/scan-page')
def scan_page():
    return render_template('scan.html')

# Route untuk deteksi real-time dari kamera
@app.route('/scan_frame', methods=['POST'])
def scan_frame():
    data = request.get_json()
    img_data = data['image'].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    
    # Jalankan Deteksi
    results = model(img)
    df = results.pandas().xyxy[0]
    
    detections = []
    for _, row in df.iterrows():
        class_name = row['name']
        info = DISEASE_INFO.get(class_name, {'nama': class_name, 'color': 'gray', 'saran': ''})
        detections.append({
            'nama': info['nama'],
            'confidence': f"{row['confidence']*100:.1f}%",
            'color': info['color'],
            'saran': info['saran'],
            # Koordinat box untuk gambar real-time
            'box': [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
        })

    return jsonify({'detections': detections})

# Route untuk upload file (Tetap seperti sebelumnya)
@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        filename = secure_filename(uploaded_file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(input_path)
        
        results = model(input_path)
        df = results.pandas().xyxy[0]
        
        detections = []
        for _, row in df.iterrows():
            class_name = row['name']
            info = DISEASE_INFO.get(class_name, {'nama': class_name, 'color': 'gray', 'saran': ''})
            detections.append({'nama': info['nama'], 'confidence': f"{row['confidence']*100:.1f}%", 'color': info['color'], 'saran': info['saran']})
        
        results.render()
        result_filename = 'res_' + filename
        Image.fromarray(results.ims[0]).save(os.path.join(RESULT_FOLDER, result_filename))
        
        return render_template('upload.html', original=filename, result=result_filename, detections=detections)
    return redirect(url_for('upload_page'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)