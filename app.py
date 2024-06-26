import streamlit as st
import numpy as np
import librosa
import plotly.express as px
from tensorflow import keras
from keras.models import load_model
from streamlit_mic_recorder import mic_recorder
import os

model = load_model('classifier.h5')

SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109

def process_audio(audio_file_path):
    X_test = []
    
    audio, _ = librosa.load(audio_file_path, sr=SAMPLE_RATE, duration=DURATION)

    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

    X_test.append(mel_spectrogram)
    X_test = np.array(X_test)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    return y_pred, audio

def main():
    st.title('Deep:blue[Fake] Audio Classifier :sparkles:')
    st.subheader('', divider='rainbow')


    st.subheader("Record the voice for DeepFake:")
    audio = mic_recorder(start_prompt="⏺️", stop_prompt="⏹️", key='recorder')

    if audio:
        temp_file_path = "temp_audio.flac"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(audio['bytes'])

        prediction, audio_clip = process_audio(temp_file_path)

        
        os.remove(temp_file_path)


        col1, col2, col3 = st.columns(3)
        with col1:
            st.text(f"Prediction: {prediction[0][0]}")
            if prediction[0][0] == 1:
                st.write('Prediction: Fake')
            else:
                st.write('Prediction: Real')

        with col2:
            st.info("Your uploaded audio is below")
            st.audio(audio['bytes'])

            fig = px.line(x=list(range(len(audio_clip))), y=audio_clip)
            fig.update_layout(
                title="Waveform plot",
                xaxis_title="Time",
                yaxis_title="Amplitude"
            )
            st.plotly_chart(fig)

        with col3:
            st.info("Disclaimer")
            st.warning("These classification or detection mechanisms are not always accurate. They should be considered as a strong signal and not the ultimate decision makers.")
    
    st.subheader("Upload your Call Recording:")
    uploaded_file = st.file_uploader("", type=['flac'])

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

            fig = px.line(x=list(range(len(audio_clip))), y=audio_clip)
            fig.update_layout(
                title="Waveform plot",
                xaxis_title="Time",
                yaxis_title="Amplitude"
            )
            st.plotly_chart(fig)

if __name__ == '__main__':
    main()
