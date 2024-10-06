from audio_processing import load_audio, create_spectrogram
from Cnn import build_cnn_rnn_model
from training import train_model
from realtime import process_realtime_audio
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Path to the datasets directory
base_dir = r'C:\Users\siddh\OneDrive\Desktop\chord_recognition\Datasets'

# Initialize lists to hold spectrograms and labels
spectrograms = []
labels = []

# Function to recursively find all .wav files in the directory and its subdirectories
def find_audio_files(directory):
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.wav'):
                file_path = os.path.join(root, file_name)
                yield file_path, root.split(os.sep)[-2]  # Use second last folder name as label

# Loop through the dataset folder and collect all .wav files and their labels
for file_path, label in find_audio_files(base_dir):
    print(f"Loading file: {file_path}")  # Print the file being loaded

    # Load the audio file
    audio, sr = load_audio(file_path)
    if audio is not None:
        # Process the audio and create a spectrogram
        spectrogram = create_spectrogram(audio, sr)
        spectrograms.append(spectrogram)

        # Print the shape of the spectrogram for debugging
        print(f"Processed {file_path}, Spectrogram shape: {spectrogram.shape}")

        # Append the corresponding label
        labels.append(label)

# Check if spectrograms and labels are loaded
if len(spectrograms) == 0 or len(labels) == 0:
    raise ValueError("No valid audio files or labels found in the dataset directory.")

# Convert the list of spectrograms into a numpy array for training
spectrograms = np.array(spectrograms)

# Convert labels to one-hot encoded vectors
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Ensure that encoded_labels is not empty before applying to_categorical
if encoded_labels.size == 0:
    raise ValueError("No labels were encoded. Please check the dataset.")

one_hot_labels = to_categorical(encoded_labels)

# Determine the number of classes
num_classes = one_hot_labels.shape[1]

# Build the CNN-RNN model, passing the correct number of classes
input_shape = (128, 128, 1)  # Adjust based on your actual spectrogram shape
model = build_cnn_rnn_model(input_shape, num_classes)

# Train the model
train_model(model, spectrograms, one_hot_labels)

# Print completion message
print("All audio files loaded, processed, and the model is trained.")

# Real-time chord recognition (Uncomment when needed)
# process_realtime_audio(model)
# After training
print("Saving the model...")
model.save('chord_recognition_model.h5')
print("Model saved successfully.")
