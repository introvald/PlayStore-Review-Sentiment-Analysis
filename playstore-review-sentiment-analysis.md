# Analisis Sentimen Ulasan Play Store (Bahasa Indonesia)

Dataset diperoleh dari hasil web scraping ulasan aplikasi di Google Play Store menggunakan `google-play-scraper`, disimpan dalam `ulasan_aplikasi.csv`. Setelah proses pembersihan (menghapus data kosong/`NaN` dan duplikat), data diberi label sentimen secara otomatis menggunakan pendekatan **lexicon-based** (kamus kata positif dan negatif berbahasa Indonesia).

Dataset mencakup 3 kelas sentimen: `positive`, `neutral`, `negative`.

| Kelas | Jumlah Data (sebelum SMOTE) |
| --- | --- |
| Negative | 5.912 |
| Positive | 3.553 |
| Neutral | 2.335 |

Karena distribusi kelas tidak seimbang, dilakukan **oversampling dengan SMOTE** sehingga setiap kelas memiliki 5.912 data (total ±17.736 data), kemudian dibagi dengan rasio 80:20 untuk data train dan test.

## Tahapan Preprocessing

Teks ulasan melalui pipeline preprocessing lengkap: cleaning (menghapus mention, hashtag, URL, angka, dan tanda baca), case folding, normalisasi kata gaul/slang (kamus slangwords kustom), tokenisasi, stopword removal (bahasa Indonesia & Inggris), hingga digabung kembali menjadi kalimat bersih (`text_akhir`).

## Arsitektur Model

Tiga skema eksperimen dibangun dan dibandingkan menggunakan kombinasi arsitektur deep learning dan teknik ekstraksi fitur yang berbeda:

| Skema | Arsitektur | Ekstraksi Fitur | Split Data |
| --- | --- | --- | --- |
| 1 | LSTM | TF-IDF | 80:20 |
| 2 | GRU | TF-IDF | 80:20 |
| 3 | CNN (Conv1D) | FastText Embedding | 80:20 |

- Skema 1 & 2 menggunakan representasi TF-IDF (3.000 fitur) yang di-reshape menjadi input sekuensial untuk layer LSTM/GRU, dengan regularisasi L2 dan Dropout untuk mencegah overfitting.
- Skema 3 menggunakan embedding kata yang dilatih sendiri dengan **FastText** (dimensi 100), lalu diproses melalui `Conv1D` + `GlobalMaxPooling1D`.
- Semua model dilatih hingga 50 epoch dengan optimizer Adam dan `EarlyStopping` (memantau `val_loss`) untuk menghentikan training saat performa validasi tidak lagi membaik.

## Hasil Pelatihan

| Skema | Model | Akurasi Train | Akurasi Test |
| --- | --- | --- | --- |
| 1 | LSTM + TF-IDF | 94.71% | 91.04% |
| 2 | GRU + TF-IDF | 93.68% | 89.80% |
| 3 | CNN + FastText | 97.46% | **93.66%** |

Model **CNN + FastText (Skema 3)** menunjukkan performa terbaik dengan akurasi test tertinggi sekaligus konvergensi yang lebih stabil dibanding dua skema lainnya.

## Format Model

Ketiga model disimpan dalam format **`.h5`** (`model_lstm.h5`, `model_gru.h5`, `model_cnn.h5`), beserta artefak pendukungnya (`tfidf_lstm.pkl`, `tfidf_gru.pkl`, `tokenizer_cnn.pkl`, `label_encoder.pkl`) menggunakan `pickle`, untuk digunakan kembali pada tahap inferensi (`inference.ipynb`).
