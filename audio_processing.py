import librosa
import numpy as np
import matplotlib.pyplot as plt
# audio_processing.py


def load_audio(file_path, sr=22050):
    try:
        # Load the audio file using librosa
        audio, sample_rate = librosa.load(file_path, sr=sr)
        return audio, sample_rate
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None



def create_spectrogram(audio, sr):
    stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
    return spectrogram

def extract_mfcc(audio, sr, n_mfcc=13):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

def plot_spectrogram(spectrogram, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()
