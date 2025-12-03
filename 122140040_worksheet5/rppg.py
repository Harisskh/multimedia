import cv2
import numpy as np
import mediapipe as mp
from scipy import signal
import time
from collections import deque

class RealTimeRPPG:
    """
    Kelas utama untuk menjalankan sistem rPPG secara real-time.
    Kelas ini  akuisisi video, deteksi wajah, pemrosesan sinyal,
    dan visualisasi data dalam webcam overlay.
    """
    
    def __init__(self, buffer_size=150, fps=30):
        """
        Inisialisasi parameter awal sistem.
        
        Args:
            buffer_size (int): Panjang antrean data (sliding window). 
                               150 frame setara dengan 5 detik pada 30 FPS.
            fps (int): Estimasi frame rate kamera.
        """
        self.buffer_size = buffer_size
        self.fps = fps
        
        # Buffer (antrean memori) untuk menyimpan data sinyal mentah dan hasil filter
        # deque digunakan agar data lama otomatis terbuang saat penuh (konsep Sliding Window)
        self.signal_buffer = deque(maxlen=buffer_size)
        self.filtered_buffer = deque(maxlen=buffer_size)
        
        # Konfigurasi MediaPipe Face Mesh untuk deteksi wajah
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,                # Hanya mendeteksi 1 wajah
            refine_landmarks=False,         # Tidak perlu deteksi iris mata (biar ringan)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Daftar Indeks Landmark Wajah untuk Region of Interest (ROI)
        # Kita memilih area Pipi dan Dahi karena kaya pembuluh darah kapiler
        # dan minim gangguan gerakan otot (seperti mulut).
        self.roi_indices = [
            330, 347, 346, 345, 348, 349, # Pipi Kiri
            101, 118, 117, 116, 119, 120, # Pipi Kanan
            10, 151, 9, 8, 107, 69, 108   # Dahi
        ]

        # Variabel untuk menyimpan nilai BPM saat ini
        self.current_bpm = 0.0
        
        # Counter untuk menghitung berapa frame wajah tidak terdeteksi
        self.no_face_frames = 0
        
        # Variabel untuk menghitung FPS (Frame per Second) agar stabil
        self.frame_counter = 0
        self.fps_start_time = time.time()
        self.display_fps = 0 # Nilai FPS yang akan ditampilkan di layar

    def get_roi_average(self, frame, landmarks):
        """
        Menghitung rata-rata warna RGB dari area kulit wajah (ROI).
        
        Fungsi ini mengambil koordinat titik wajah dari MediaPipe, membuat mask (poligon),
        dan menghitung rata-rata piksel di dalam area tersebut. Teknik ini disebut
        Spatial Averaging untuk mengurangi noise sensor kamera.

        Args:
            frame: Gambar frame video saat ini.
            landmarks: Objek hasil deteksi landmark dari MediaPipe.

        Returns:
            numpy array: Nilai rata-rata [R, G, B] dari area ROI.
        """
        h, w, _ = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        points = []
        
        # Mengonversi indeks landmark menjadi koordinat piksel (x, y)
        for idx in self.roi_indices:
            pt = landmarks.landmark[idx]
            x = int(pt.x * w)
            y = int(pt.y * h)
            points.append([x, y])
            
        # Jika titik ditemukan, buat area tertutup (Convex Hull)
        if points:
            hull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(mask, hull, 255)
            
            # Hitung rata-rata warna hanya pada area mask yang putih
            mean_val = cv2.mean(frame, mask=mask)
            
            # OpenCV urutannya BGR, kita balik jadi RGB agar sesuai algoritma POS
            return np.array([mean_val[2], mean_val[1], mean_val[0]])
        
        return None

    def bandpass_filter(self, data):
        """
        Menerapkan Filter Bandpass Butterworth.
        
        Berfungsi untuk membuang frekuensi sinyal yang tidak mungkin berasal dari detak jantung.
        Rentang yang diizinkan: 0.7 Hz (42 BPM) sampai 4.0 Hz (240 BPM).
        
        Args:
            data: List/Array sinyal mentah.
        Returns:
            Array sinyal yang sudah difilter (bersih dari noise frekuensi rendah/tinggi).
        """
        if len(data) < 30: return data # Belum cukup data untuk difilter
        
        nyquist = 0.5 * self.fps
        low = 0.7 / nyquist
        high = 4.0 / nyquist
        
        # Filter Orde 3
        b, a = signal.butter(3, [low, high], btype='band')
        return signal.lfilter(b, a, data)

    def process_pos_algorithm(self, rgb_data):
        """
        Implementasi Algoritma POS (Plane-Orthogonal-to-Skin).
        
        Ini adalah inti matematika rPPG. Algoritma ini memisahkan sinyal darah (yang kita cari)
        dari sinyal gerakan dan perubahan cahaya (noise) dengan memproyeksikan kanal warna RGB
        ke bidang ortogonal.

        Args:
            rgb_data: Kumpulan data RGB dari buffer.
        Returns:
            Array sinyal satu dimensi (Pulse Wave).
        """
        rgb_data = np.array(rgb_data)
        if len(rgb_data) == 0: return np.array([])
        
        # Normalisasi temporal
        mean_rgb = np.mean(rgb_data, axis=0)
        norm_rgb = rgb_data / (mean_rgb + 1e-6)
        
        r, g, b = norm_rgb[:, 0], norm_rgb[:, 1], norm_rgb[:, 2]
        
        # Proyeksi matematis POS
        s1 = g - b
        s2 = g + b - (2 * r)
        
        # Tuning alpha (standar deviasi)
        alpha = np.std(s1) / (np.std(s2) + 1e-6)
        
        # Sinyal gabungan (H)
        h = s1 + (alpha * s2)
        
        # Detrending (mengurangi rata-rata agar sinyal berpusat di 0)
        h = h - np.mean(h)
        return h

    def calculate_bpm(self):
        """
        Menghitung nilai BPM (Beats Per Minute) menggunakan FFT.
        
        Fungsi ini mengubah sinyal gelombang (Domain Waktu) menjadi frekuensi (Domain Frekuensi)
        menggunakan Fast Fourier Transform (FFT) untuk mencari frekuensi detak jantung dominan.

        Returns:
            tuple: (Nilai BPM, Sinyal yang sudah difilter untuk grafik)
        """
        if len(self.signal_buffer) < self.buffer_size:
            return 0.0, []
            
        raw_rgb = list(self.signal_buffer)
        
        # 1. Ekstraksi Sinyal (POS)
        pos_signal = self.process_pos_algorithm(raw_rgb)
        
        # 2. Filtering (Bandpass)
        filtered_signal = self.bandpass_filter(pos_signal)
        
        # Simpan sinyal untuk ditampilkan di grafik nanti
        self.filtered_buffer.extend(filtered_signal[-int(len(filtered_signal)/2):])

        # 3. FFT (Transformasi Fourier)
        n_samples = len(filtered_signal)
        fft_values = np.fft.rfft(filtered_signal)
        freqs = np.fft.rfftfreq(n_samples, 1.0 / self.fps)
        
        # Cari frekuensi dominan di rentang 0.7 - 4.0 Hz
        valid_idx = np.where((freqs >= 0.7) & (freqs <= 4.0))
        valid_freqs = freqs[valid_idx]
        valid_fft = np.abs(fft_values[valid_idx])
        
        if len(valid_fft) == 0:
            return 0.0, filtered_signal
            
        # Ambil puncak tertinggi (Peak Detection)
        max_idx = np.argmax(valid_fft)
        dominant_freq = valid_freqs[max_idx]
        
        # Konversi Hz ke BPM (1 Hz = 60 BPM)
        return dominant_freq * 60.0, filtered_signal

    def create_dashboard(self, width, height):
        """
        Membuat tampilan antarmuka (Dashboard) di bagian bawah video.
        Dashboard ini berisi teks BPM, Status Buffer, FPS, dan Grafik Gelombang.

        Args:
            width: Lebar frame video.
            height: Tinggi area dashboard (misal 150 piksel).
        Returns:
            numpy array: Gambar dashboard hitam yang siap digabung dengan video.
        """
        dashboard = np.zeros((height, width, 3), dtype=np.uint8)
        
        # --- BAGIAN KIRI: TEKS INFORMASI ---
        
        # Logika: Jika wajah hilang, tampilkan peringatan
        if self.no_face_frames > 10: 
            cv2.putText(dashboard, "WAJAH TIDAK", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(dashboard, "TERDETEKSI", (20, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.current_bpm = 0.0
            
        # Logika: Jika buffer belum penuh, tampilkan status Kalibrasi
        elif len(self.signal_buffer) < self.buffer_size:
            cv2.putText(dashboard, "KALIBRASI...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(dashboard, f"Data: {len(self.signal_buffer)}/{self.buffer_size}", (20, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Logika: Jika normal, tampilkan BPM
        else:
            bpm_color = (0, 255, 0) # Hijau (Normal)
            if self.current_bpm < 50 or self.current_bpm > 120:
                 bpm_color = (0, 0, 255) # Merah (Peringatan)
            
            cv2.putText(dashboard, f"BPM: {int(self.current_bpm)}", (20, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, bpm_color, 3)
            
        # Tampilkan FPS (Pojok Kiri Bawah)
        cv2.putText(dashboard, f"FPS: {self.display_fps}", (20, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # --- BAGIAN KANAN: GRAFIK GELOMBANG (WAVEFORM) ---
        graph_start_x = 280 # Koordinat X mulai grafik
        graph_width = width - graph_start_x
        
        # Garis pembatas vertikal
        cv2.line(dashboard, (graph_start_x - 20, 10), (graph_start_x - 20, height - 10), (50, 50, 50), 2)

        # Gambar grafik hanya jika ada data dan wajah terdeteksi
        if self.no_face_frames <= 10 and len(self.filtered_buffer) > 10:
            data_to_plot = list(self.filtered_buffer)[-150:] 
            min_val = min(data_to_plot)
            max_val = max(data_to_plot)
            
            points = []
            # Normalisasi data agar pas di dalam kotak grafik
            if max_val - min_val != 0:
                for i, val in enumerate(data_to_plot):
                    x = int(graph_start_x + (i / len(data_to_plot)) * graph_width)
                    y = int(height - 20 - ((val - min_val) / (max_val - min_val) * (height - 40)))
                    points.append((x, y))

                if len(points) > 1:
                    cv2.polylines(dashboard, [np.array(points)], False, (0, 255, 255), 2)
        
        # Judul Grafik
        title_text = "Pulse Wave" if self.no_face_frames <= 10 else "No Signal"
        cv2.putText(dashboard, title_text, (graph_start_x, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                   
        return dashboard

    def run(self):
        """
        Fungsi utama (Main Loop).
        Membuka kamera, membaca frame per frame, dan menjalankan pipeline pemrosesan.
        """
        cap = cv2.VideoCapture(0)
        
        # Paksa resolusi kamera ke 640x480 agar performa FPS stabil
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        last_bpm_update = time.time()
        print("Sistem berjalan. Tekan 'q' untuk keluar.")
        
        while cap.isOpened():
            # --- UPDATE FPS (METODE 1 DETIK) ---
            # Menghitung frame setiap detik agar angka FPS di layar tidak loncat-loncat
            self.frame_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.display_fps = self.frame_counter
                self.frame_counter = 0
                self.fps_start_time = time.time()

            ret, frame = cap.read()
            if not ret: break
            
            # Balik video horizontal (Mirroring) agar natural
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = frame.shape
            
            # 1. DETEKSI WAJAH & ROI
            
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                self.no_face_frames = 0 
                for face_landmarks in results.multi_face_landmarks:
                    # Ambil rata-rata warna dari Pipi dan Dahi
                    roi_avg = self.get_roi_average(frame, face_landmarks)
                    if roi_avg is not None:
                        self.signal_buffer.append(roi_avg)
                    
                    # Visualisasi: Gambar Kotak Hijau di area ROI
                    roi_points = []
                    for idx in self.roi_indices:
                        pt = face_landmarks.landmark[idx]
                        roi_points.append((int(pt.x * w), int(pt.y * h)))
                    if roi_points:
                        x_r, y_r, w_r, h_r = cv2.boundingRect(np.array(roi_points))
                        cv2.rectangle(frame, (x_r, y_r), (x_r + w_r, y_r + h_r), (0, 255, 0), 2)
            else:
                # Jika wajah tidak terdeteksi
                self.no_face_frames += 1
                if self.no_face_frames > 10:
                    # Reset buffer jika wajah hilang terlalu lama
                    self.signal_buffer.clear()
                    self.filtered_buffer.clear()
                    self.current_bpm = 0.0

            # 2. UPDATE BPM SETIAP 1 DETIK
            if time.time() - last_bpm_update > 1.0:
                # Syarat: Buffer harus penuh dan wajah harus terdeteksi
                if len(self.signal_buffer) == self.buffer_size and self.no_face_frames <= 10:
                    new_bpm, filtered = self.calculate_bpm()
                    if len(filtered) > 0:
                        self.filtered_buffer = deque(filtered)
                    
                    # Filtering hasil BPM agar tidak nol atau outlier
                    if new_bpm > 40: 
                        if self.current_bpm == 0: self.current_bpm = new_bpm
                        else: self.current_bpm = 0.9 * self.current_bpm + 0.1 * new_bpm
                last_bpm_update = time.time()

            # 3. RENDER TAMPILAN AKHIR (STACKING)
            # Membuat dashboard hitam lalu menumpuknya di bawah video
            dashboard = self.create_dashboard(w, 150)
            try:
                final_display = np.vstack([frame, dashboard])
                cv2.imshow('rPPG Dashboard', final_display)
            except ValueError: pass

            # Tekan 'q' untuk berhenti
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app = RealTimeRPPG()
        app.run()
    except Exception as e:
        print(f"Error: {e}")