import os
import sys
from unittest.mock import MagicMock

# Mocking PyAudio agar kompatibel di Streamlit Cloud
sys.modules['pyaudio'] = MagicMock()

import io
import base64
import tempfile
import soundfile as sf
import librosa
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from BeatNet.BeatNet import BeatNet

st.set_page_config(page_title="Multi-Track Smart Click Player", page_icon="🎛️", layout="wide")
st.title("🎛️ Multi-Track Smart Click Player (Ala Moises)")
st.write("Lagu dan *Click Track* diputar di channel terpisah namun tersinkronisasi secara *real-time*.")

# Fungsi pembantu untuk mengonversi numpy audio ke Base64 Data URI
def get_audio_base64(audio_array, sr):
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sr, format='WAV')
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:audio/wav;base64,{b64}"

# Upload File Audio
uploaded_file = st.file_uploader("Unggah Lagu (MP3 / WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🚀 Hasilkan Smart Click (Separated Track)"):
        with st.spinner("BeatNet AI sedang mengekstrak ketukan lagu..."):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_audio_path = tmp_file.name

            try:
                # 1. Analisis AI BeatNet
                estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
                output = estimator.process(temp_audio_path)
                
                y, sr = librosa.load(temp_audio_path, sr=None)
                audio_length = len(y)

                # 2. Pisahkan Timestamp Downbeat (Accent) & Regular Beat
                downbeat_times = output[output[:, 1] == 1, 0]
                regular_beat_times = output[output[:, 1] != 1, 0]

                # 3. Buat Hanya Track Klik Saja (Lagu Asli Tetap Bersih)
                clicks_downbeat = librosa.clicks(times=downbeat_times, sr=sr, length=audio_length, click_freq=1200.0)
                clicks_regular = librosa.clicks(times=regular_beat_times, sr=sr, length=audio_length, click_freq=800.0)
                
                click_track_only = clicks_downbeat + (clicks_regular * 0.7)
                # Normalisasi audio klik
                click_track_only = click_track_only / np.max(np.abs(click_track_only))

                # 4. Konversi Kedua Track ke Format Base64 agar BIsa Dibaca HTML/JS Player
                music_b64 = get_audio_base64(y, sr)
                click_b64 = get_audio_base64(click_track_only, sr)

                st.success("✅ Multi-Track Player Siap!")

                # 5. Player HTML5 + JavaScript Kustom untuk Multi-Channel Playback
                html_player_code = f"""
                <div style="background-color: #0e1117; color: white; padding: 20px; border-radius: 12px; font-family: sans-serif; border: 1px solid #30363d;">
                    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
                        <button id="playBtn" onclick="togglePlay()" style="background-color: #ff4b4b; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 16px;">
                            ▶ Play Both
                        </button>
                        <span id="timeDisplay" style="font-family: monospace; font-size: 14px;">00:00 / 00:00</span>
                    </div>

                    <!-- Progress/Seek Bar -->
                    <input type="range" id="seekBar" min="0" max="100" value="0" step="0.1" oninput="seekAudio(this.value)" style="width: 100%; margin-bottom: 20px; cursor: pointer;">

                    <!-- Controls Grid -->
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <!-- Track 1: Lagu Asli -->
                        <div style="background-color: #161b22; padding: 15px; border-radius: 8px;">
                            <label style="font-weight: bold; display: block; margin-bottom: 8px;">🎵 Volume Lagu Asli</label>
                            <input type="range" id="musicVol" min="0" max="1" step="0.05" value="1" oninput="setMusicVol(this.value)" style="width: 100%;">
                        </div>

                        <!-- Track 2: Click Track (Metronom) -->
                        <div style="background-color: #161b22; padding: 15px; border-radius: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <label style="font-weight: bold;">🥁 Volume Click Track</label>
                                <button onclick="toggleMuteClick()" id="muteBtn" style="background-color: #21262d; color: white; border: 1px solid #30363d; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px;">
                                    Mute
                                </button>
                            </div>
                            <input type="range" id="clickVol" min="0" max="1" step="0.05" value="0.8" oninput="setClickVol(this.value)" style="width: 100%;">
                        </div>
                    </div>
                </div>

                <script>
                    const music = new Audio("{music_b64}");
                    const click = new Audio("{click_b64}");
                    
                    const playBtn = document.getElementById("playBtn");
                    const seekBar = document.getElementById("seekBar");
                    const timeDisplay = document.getElementById("timeDisplay");
                    const clickVolInput = document.getElementById("clickVol");
                    
                    let isMuted = false;
                    let lastClickVol = 0.8;

                    music.volume = 1;
                    click.volume = 0.8;

                    // Update seekbar & durasi waktu
                    music.ontimeupdate = () => {{
                        if (!isNaN(music.duration)) {{
                            seekBar.value = (music.currentTime / music.duration) * 100;
                            timeDisplay.innerText = formatTime(music.currentTime) + " / " + formatTime(music.duration);
                        }}
                    }};

                    function formatTime(seconds) {{
                        const mins = Math.floor(seconds / 60);
                        const secs = Math.floor(seconds % 60);
                        return `${{mins.toString().padStart(2, '0')}}:${{secs.toString().padStart(2, '0')}}`;
                    }}

                    function togglePlay() {{
                        if (music.paused) {{
                            click.currentTime = music.currentTime; // Sync frame sebelum play
                            music.play();
                            click.play();
                            playBtn.innerText = "⏸ Pause";
                            playBtn.style.backgroundColor = "#238636";
                        }} else {{
                            music.pause();
                            click.pause();
                            playBtn.innerText = "▶ Play Both";
                            playBtn.style.backgroundColor = "#ff4b4b";
                        }}
                    }}

                    function seekAudio(value) {{
                        const targetTime = (value / 100) * music.duration;
                        music.currentTime = targetTime;
                        click.currentTime = targetTime; // Menjaga sinkronisasi saat digeser/seek
                    }}

                    function setMusicVol(val) {{
                        music.volume = val;
                    }}

                    function setClickVol(val) {{
                        click.volume = val;
                        if(val > 0) isMuted = false;
                    }}

                    function toggleMuteClick() {{
                        const muteBtn = document.getElementById("muteBtn");
                        if (!isMuted) {{
                            lastClickVol = click.volume;
                            click.volume = 0;
                            clickVolInput.value = 0;
                            muteBtn.innerText = "Unmute";
                            muteBtn.style.backgroundColor = "#da3633";
                            isMuted = true;
                        }} else {{
                            click.volume = lastClickVol;
                            clickVolInput.value = lastClickVol;
                            muteBtn.innerText = "Mute";
                            muteBtn.style.backgroundColor = "#21262d";
                            isMuted = false;
                        }}
                    }}
                </script>
                """

                # Render HTML Player di Streamlit
                components.html(html_player_code, height=220)

                # Fitur Ekstra: Opsi Download Track Secara Terpisah
                st.subheader("📥 Opsi Unduh File Terpisah")
                col1, col2 = st.columns(2)

                # Download Click Track Only
                click_buffer = io.BytesIO()
                sf.write(click_buffer, click_track_only, sr, format='WAV')
                with col1:
                    st.download_button(
                        label="Download Click Track Saja (.wav)",
                        data=click_buffer.getvalue(),
                        file_name="click_track_only.wav",
                        mime="audio/wav"
                    )

            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
