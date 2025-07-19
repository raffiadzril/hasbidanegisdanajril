# Analisis Komentar Judi Online dengan NLP

Sistem analisis untuk mendeteksi komentar yang berkaitan dengan judi online menggunakan Natural Language Processing (NLP) dan Machine Learning.

## 🎯 Fitur Utama

1. **Deteksi Otomatis**: Menggunakan AI untuk mendeteksi komentar judi online tanpa regex
2. **Analisis Sentiment**: Menganalisis sentiment dari komentar yang terdeteksi
3. **Visualisasi**: Grafik dan word cloud untuk memahami pola
4. **Laporan Detail**: CSV output dengan confidence score dan features

## 📋 Cara Penggunaan

### 1. Install Dependencies
```bash
# Jalankan file batch untuk install semua library
install_dependencies.bat

# Atau install manual:
pip install -r requirements.txt
```

### 2. Jalankan Analisis
```bash
python analisis_judi_nlp.py
```

### 3. Lihat Hasil
- `hasil_analisis_judi.csv` - Semua hasil analisis
- `komentar_judi_terdeteksi.csv` - Hanya komentar judi yang terdeteksi
- `analisis_judi_online.png` - Grafik analisis
- `wordcloud_judi.png` - Word cloud komentar judi

## 🤖 Cara Kerja Sistem

### 1. Preprocessing
- Membersihkan teks (URL, mention, punctuation)
- Normalisasi teks (lowercase, whitespace)
- Tokenisasi

### 2. Feature Extraction
- **Gambling Keywords**: Menghitung kata kunci judi (slot, gacor, maxwin, dll)
- **Sentiment Analysis**: Menganalisis polaritas dan subjektivitas
- **Contact Info**: Mendeteksi pola nomor HP, WhatsApp, dll
- **Promotional Words**: Mendeteksi kata-kata promosi
- **Text Statistics**: Panjang teks, jumlah kata

### 3. Machine Learning Model
- **TF-IDF Vectorizer**: Mengubah teks menjadi features numerik
- **Naive Bayes Classifier**: Model klasifikasi untuk prediksi
- **Pipeline**: Kombinasi preprocessing dan model

### 4. Kata Kunci Judi yang Dideteksi
```
- slot, gacor, maxwin, scatter, bonus
- deposit, withdraw, saldo, jackpot, rtp
- pragmatic, pg soft, gates of olympus
- sweet bonanza, starlight princess
- situs judi, bandar, togel, casino
- poker online, domino, betting, taruhan
- dan masih banyak lagi...
```

## 📊 Output Analisis

### File `hasil_analisis_judi.csv`
Kolom:
- `author`: Nama author komentar
- `original_comment`: Komentar asli
- `is_gambling_prediction`: Prediksi (True/False)
- `confidence`: Confidence score (0-1)
- `gambling_keywords_found`: Jumlah kata kunci judi
- `has_contact_info`: Ada info kontak atau tidak
- `sentiment_polarity`: Nilai sentiment (-1 sampai 1)

### Visualisasi
1. **Pie Chart**: Distribusi komentar judi vs non-judi
2. **Histogram**: Distribusi confidence score
3. **Scatter Plot**: Sentiment vs Confidence
4. **Word Cloud**: Kata-kata yang sering muncul di komentar judi

## 🎯 Akurasi dan Performance

Sistem menggunakan kombinasi:
- **Rule-based features** untuk akurasi tinggi
- **Machine learning** untuk adaptabilitas
- **Multiple features** untuk robustness

Performance metrics akan ditampilkan setelah training:
- Precision: Seberapa akurat prediksi positif
- Recall: Seberapa baik mendeteksi semua kasus judi
- F1-Score: Keseimbangan precision dan recall

## 🔧 Kustomisasi

### Menambah Kata Kunci Baru
Edit list `gambling_keywords` di dalam class `JudiOnlineDetector`:
```python
self.gambling_keywords = [
    'kata_baru_1', 'kata_baru_2', 
    # tambahkan kata kunci baru di sini
]
```

### Mengubah Threshold Deteksi
Edit kondisi dalam method `create_training_data()`:
```python
# Ubah threshold dari 2 menjadi nilai lain
if features['gambling_keywords_count'] >= 2:
    is_gambling = True
```

## 📝 Catatan Penting

1. **Data Training**: Sistem membuat training data otomatis berdasarkan rule-based labeling
2. **Bahasa**: Dioptimalkan untuk bahasa Indonesia dan campuran dengan Inggris
3. **Update Model**: Model dapat di-retrain dengan data baru untuk meningkatkan akurasi
4. **False Positive**: Beberapa komentar mungkin ter-flag salah, review manual disarankan

## 🚀 Tips Penggunaan

1. **Review Manual**: Selalu review hasil dengan confidence score rendah
2. **Update Keywords**: Tambah kata kunci baru sesuai trend terbaru
3. **Batch Processing**: Untuk dataset besar, bisa diproses dalam batch
4. **Monitoring**: Pantau performa model secara berkala

## ⚠️ Disclaimer

Sistem ini adalah alat bantu analisis. Hasil akhir sebaiknya dikombinasikan dengan review manual untuk akurasi maksimal.
