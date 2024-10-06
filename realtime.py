import sounddevice as sd
from audio_processing import create_spectrogram

def process_realtime_audio(model, duration=5, sr=22050):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Processing audio...")

    audio = np.squeeze(audio)  # Remove single-dimensional entries
    spectrogram = create_spectrogram(audio, sr)

    spectrogram = np.expand_dims(spectrogram, axis=(0, -1))  # Reshape for model

    # Predict the chord
    prediction = model.predict(spectrogram)
    predicted_chord = np.argmax(prediction)
    print(f'Predicted Chord: {predicted_chord}')
