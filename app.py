import os
import tempfile
import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from BeatNet.BeatNet import BeatNet

st.set_page_config(page_title="AI Smart Click Metronome", page_icon="🥁")
st.title("🥁 Smart Click Metronome (Powered by BeatNet AI)")
st.write("Mendukung deteksi **Downbeat (Ketukan Ke-1)** & **Tempo Dinamis** setara Moises.")

# 1. Upload File Audio
uploaded_file = st.file_uploader("Unggah Lagu (MP3 / WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    # Pengaturan Volume Click Track
    st.subheader("⚙️ Pengaturan Click Track")
    downbeat_vol = st.slider("Volume Ketukan Pertama (Accent/Downbeat)", 0.0, 1.5, 1.0, 0.1)
    beat_vol = st.slider("Volume Ketukan Biasa (Sub-beats)", 0.0, 1.5, 0.6, 0.1)

    if st.button("🚀 Jalankan BeatNet AI & Sync Click"):
        with st.spinner("BeatNet AI sedang menganalisis struktur lagu & downbeats..."):
            
            # Simpan file sementara ke disk (BeatNet membutuhkan file path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_audio_path = tmp_file.name

            try:
                # 2. Inisialisasi Model BeatNet (Mode Offline)
                # Model '1' adalah model offline DBN terbaik untuk musik
                estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
                
                # 3. Proses Analisis Audio
                # Output berupa array 2D: [Timestamp (detik), Jenis Ketukan (1=Downbeat, 2/3/4=Beats)]
                output = estimator.process(temp_audio_path)
                
                # Load audio asli untuk digabungkan
                y, sr = librosa.load(temp_audio_path, sr=None)
                audio_length = len(y)

                # 4. Pisahkan Stempel Waktu Downbeat dan Ketukan Biasa
                downbeat_times = output[output[:, 1] == 1, 0]  # Ketukan ke-1
                regular_beat_times = output[output[:, 1] != 1, 0]  # Ketukan 2, 3, 4

                # 5. Sintesis Gelombang Audio Klik
                # Downbeat diberi nada lebih tinggi (1200 Hz), Ketukan biasa lebih rendah (800 Hz)
                clicks_downbeat = librosa.clicks(
                    times=downbeat_times, sr=sr, length=audio_length, click_freq=1200.0
                )
                clicks_regular = librosa.clicks(
                    times=regular_beat_times, sr=sr, length=audio_length, click_freq=800.0
                )

                # 6. Gabungkan Audio Asli + Click Accent + Click Biasa
                click_track = (clicks_downbeat * downbeat_vol) + (clicks_regular * beat_vol)
                combined_audio = y + click_track

                # Normalize agar audio tidak distorsi/clipping
                combined_audio = combined_audio / np.max(np.abs(combined_audio))

                # 7. Simpan Hasil Audio
                output_filename = "output_smart_click.wav"
                sf.write(output_filename, combined_audio, sr)

                st.success("✅ Smart Click berhasil disinkronkan!")
                st.audio(output_filename)

                # Tombol Download
                with open(output_filename, "rb") as file:
                    st.download_button(
                        label="💾 Download Lagu + Smart Click",
                        data=file,
                        file_name="lagu_smart_click.wav",
                        mime="audio/wav"
                    )

            finally:
                # Hapus file temporary
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
          
