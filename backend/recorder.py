import scipy
import numpy as np
import pyaudio

def record_audio(n_seconds: int = 5) -> str:
    """
    records audio using computer microphone
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000


    input("Press Enter to begin recording 🎤")
    print("🎤 Listening for music", end="\r")

    outfile = "microphone_sample.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    for _ in range(0, int(RATE / CHUNK * n_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()


    audio_np = np.frombuffer(b''.join(frames), dtype=np.int16)
    scipy.io.wavfile.write(outfile, RATE, audio_np)
    print(f"✅ Recording saved to {outfile}", end="\n")

    return outfile
