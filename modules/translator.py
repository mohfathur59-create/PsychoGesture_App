class SignTranslator:
    def __init__(self):
        self.current_word = ""
        self.last_prediction = None
        self.frame_count = 0
        self.threshold_frames = 20 # Butuh 20 frame stabil baru ganti huruf

    def translate_abjad(self, prediction, confidence):
        # Biar nggak gampang berubah-ubah (stabilisasi)
        if prediction == self.last_prediction:
            self.frame_count += 1
        else:
            self.frame_count = 0
            self.last_prediction = prediction

        # Kalau gerakan tangan stabil selama beberapa saat
        if self.frame_count >= self.threshold_frames and confidence > 0.7:
            return f"Terdeteksi Huruf: {prediction}"
        
        return "Mendeteksi..."

    def translate_keluhan(self, prediction):
        # Mapping hasil prediksi keluhan ke pesan yang lebih humanis
        kamus_keluhan = {
            'cemas': "User terlihat menunjukkan tanda kecemasan.",
            'bingung': "User tampak memerlukan penjelasan lebih lanjut.",
            'sedih': "User menunjukkan gestur emosi sedih.",
            'normal': "User dalam kondisi tenang."
        }
        return kamus_keluhan.get(prediction.lower(), "Gerakan tidak dikenali")