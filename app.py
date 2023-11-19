import streamlit as st
import numpy as np
import librosa
import plotly.express as px
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('classifier.h5')

# model.summary()
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109

def process_audio(audio_file):
    # Load and preprocess test data using librosa
    X_test = []
    
    audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE, duration=DURATION)

    # Extract Mel spectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure all spectrograms have the same width (time steps)
    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

    X_test.append(mel_spectrogram)

    # Convert list to numpy array
    X_test = np.array(X_test)

    # Predict using the loaded model
    y_pred = model.predict(X_test)

    # Convert probabilities to predicted classes
    y_pred_classes = np.argmax(y_pred, axis=1)

    return y_pred, audio

# Streamlit app
def main():
    st.title('Deepfake Audio Classifier')
    st.write("Upload an audio file for classification.")

    uploaded_file = st.file_uploader("Choose an audio file", type=['flac'])

    if uploaded_file is not None:
        prediction, audio_clip = process_audio(uploaded_file)

        col1, col2, col3 = st.columns(3)
        with col1:
            # st.text(prediction)
            st.header("Result")
            if prediction[0][0] == 1:
                st.write('Prediction: Fake')
            else:
                st.write('Prediction: Real')

        with col2:
            st.header("Audio file")
            st.info("Your uploaded audio is below")
            st.audio(uploaded_file)

            # create a waveform
            fig = px.line(x=list(range(len(audio_clip))), y=audio_clip)
            fig.update_layout(
                title="Waveform plot",
                xaxis_title="Time",
                yaxis_title="Amplitude"
            )
            st.plotly_chart(fig)
            
        

if __name__ == '__main__':
    main()