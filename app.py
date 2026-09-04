import os
import tempfile
import logging
import sys

# Ensure INFO+ logs are emitted to stderr so Cloud Run captures them
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

import librosa
import threading
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from tensorflow import keras

SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109
MODEL_PATH = os.environ.get("MODEL_PATH", "classifier.h5")
MODEL_GCS_URI = os.environ.get("MODEL_GCS_URI", "")


@st.cache_resource
def load_model():
    # If a GCS URI is provided, attempt to download the model into MODEL_PATH.
    if MODEL_GCS_URI and not os.path.exists(MODEL_PATH):
        try:
            from google.cloud import storage

            import hashlib
            import base64

            # Expecting a URI like gs://bucket/path/to/classifier.h5
            if MODEL_GCS_URI.startswith("gs://"):
                _, _, path = MODEL_GCS_URI.partition("gs://")
                bucket_name, _, blob_name = path.partition("/")
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.download_to_filename(MODEL_PATH)

                # Integrity checks: log size, header bytes, and compare MD5
                try:
                    blob_md5 = blob.md5_hash  # base64-encoded MD5 from GCS
                except Exception:
                    blob_md5 = None

                local_size = os.path.getsize(MODEL_PATH)
                with open(MODEL_PATH, "rb") as fh:
                    header = fh.read(8)
                    fh.seek(0)
                    local_md5 = base64.b64encode(hashlib.md5(fh.read()).digest()).decode()

                logging.info(
                    "Model downloaded: path=%s size=%d header=%s gcs_md5=%s local_md5=%s",
                    MODEL_PATH,
                    local_size,
                    header.hex(),
                    blob_md5,
                    local_md5,
                )

                if blob_md5 and local_md5 != blob_md5:
                    raise RuntimeError(
                        f"Downloaded model MD5 mismatch (gcs={blob_md5} != local={local_md5})"
                    )
                
            else:
                # Fallback: try HTTP(S) download
                import requests

                resp = requests.get(MODEL_GCS_URI, stream=True)
                resp.raise_for_status()
                with open(MODEL_PATH, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        fh.write(chunk)

                # For HTTP(S) sources compute local md5 and header
                try:
                    import hashlib, base64

                    local_size = os.path.getsize(MODEL_PATH)
                    with open(MODEL_PATH, "rb") as fh:
                        header = fh.read(8)
                        fh.seek(0)
                        local_md5 = base64.b64encode(hashlib.md5(fh.read()).digest()).decode()
                    logging.info(
                        "Model downloaded via HTTP: path=%s size=%d header=%s local_md5=%s",
                        MODEL_PATH,
                        local_size,
                        header.hex(),
                        local_md5,
                    )
                except Exception:
                    pass
        except Exception as exc:  # pragma: no cover - best-effort download
            raise RuntimeError(f"Failed to obtain model from '{MODEL_GCS_URI}': {exc}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Add classifier.h5 to the container, set MODEL_PATH, or set MODEL_GCS_URI."
        )

    # Attempt to load the Keras model and provide rich diagnostics on failure
    try:
        logging.info("Loading model from %s", MODEL_PATH)
        model = keras.models.load_model(MODEL_PATH)
        logging.info("Model loaded successfully from %s", MODEL_PATH)
        return model
    except Exception as exc:  # pragma: no cover - runtime error diagnostics
        logging.exception("Failed to load model at %s", MODEL_PATH)

        # Gather file diagnostics to aid debugging
        file_exists = os.path.exists(MODEL_PATH)
        file_size = None
        header_hex = None
        local_md5 = None

        if file_exists:
            try:
                file_size = os.path.getsize(MODEL_PATH)
            except Exception:
                logging.exception("Failed to stat model file %s", MODEL_PATH)

            try:
                with open(MODEL_PATH, "rb") as fh:
                    header = fh.read(16)
                    header_hex = header.hex()
                    fh.seek(0)
                    import hashlib, base64

                    local_md5 = base64.b64encode(hashlib.md5(fh.read()).digest()).decode()
            except Exception:
                logging.exception("Failed to read model file %s", MODEL_PATH)

        logging.error(
            "Model load diagnostics: exists=%s size=%s header=%s md5=%s",
            file_exists,
            file_size,
            header_hex,
            local_md5,
        )

        # Try to open with h5py for a clearer HDF5-specific error when available
        try:
            import h5py

            try:
                with h5py.File(MODEL_PATH, "r") as hf:
                    keys = list(hf.keys())
                logging.info("h5py opened model file; top-level keys: %s", keys)
            except Exception:
                logging.exception("h5py failed to open model file %s", MODEL_PATH)
        except Exception:
            logging.info("h5py not available in runtime; skipping HDF5 open check")

        raise RuntimeError(
            f"Unable to load model at '{MODEL_PATH}': {exc}. "
            "See instance logs for file diagnostics (exists/size/header/md5)."
        ) from exc


def process_audio(audio_source):
    temp_file_path = None

    try:
        if hasattr(audio_source, "read"):
            with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as temp_file:
                temp_file.write(audio_source.read())
                temp_file_path = temp_file.name
        else:
            temp_file_path = audio_source

        X_test = []
        audio, _ = librosa.load(temp_file_path, sr=SAMPLE_RATE, duration=DURATION)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_mels=N_MELS
        )
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
            mel_spectrogram = np.pad(
                mel_spectrogram,
                ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])),
                mode="constant",
            )
        else:
            mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

        X_test.append(mel_spectrogram)
        X_test = np.array(X_test)

        prediction = load_model().predict(X_test)
        return prediction, audio
    finally:
        if temp_file_path and temp_file_path != audio_source:
            try:
                os.remove(temp_file_path)
            except FileNotFoundError:
                pass


def main():
    # Warm the model in background to avoid blocking the frontend on cold start
    if "model_warming_started" not in st.session_state:
        st.session_state.model_warming_started = True
        st.session_state.model_ready = False
        st.session_state.model_error = ""

        def _warm():
            try:
                load_model()
                st.session_state.model_ready = True
            except Exception as e:  # pragma: no cover - runtime
                st.session_state.model_error = str(e)

        threading.Thread(target=_warm, daemon=True).start()

    if st.session_state.get("model_error"):
        st.error(f"Unable to load the model. Error: {st.session_state.get('model_error')}")
        # allow UI to remain visible so user can see instructions

    st.title("Deep:blue[Fake] Audio Classifier :sparkles:")
    st.subheader("", divider="rainbow")

    st.subheader("Record the voice for DeepFake:")
    if not st.session_state.get("model_ready"):
        with st.container():
            st.info("Model is warming up — predictions will be available shortly.")
    audio = mic_recorder(start_prompt="⏺️", stop_prompt="⏹️", key="recorder")

    if audio:
        if not st.session_state.get("model_ready"):
            st.warning("Model is still loading — try again in a moment.")
        else:
            temp_file_path = "temp_audio.flac"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(audio["bytes"])

            prediction, audio_clip = process_audio(temp_file_path)
            os.remove(temp_file_path)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.text(f"Prediction: {prediction[0][0]}")
            if prediction[0][0] == 1:
                st.write("Prediction: Fake")
            else:
                st.write("Prediction: Real")

        with col2:
            st.info("Your uploaded audio is below")
            st.audio(audio["bytes"])

            fig = px.line(x=list(range(len(audio_clip))), y=audio_clip)
            fig.update_layout(
                title="Waveform plot",
                xaxis_title="Time",
                yaxis_title="Amplitude",
            )
            st.plotly_chart(fig)

        with col3:
            st.info("Disclaimer")
            st.warning(
                "These classification or detection mechanisms are not always accurate. "
                "They should be considered as a strong signal and not the ultimate decision makers."
            )

    st.subheader("Upload your Call Recording:")
    uploaded_file = st.file_uploader("", type=["flac"])

    if uploaded_file is not None:
        if not st.session_state.get("model_ready"):
            st.warning("Model is still loading — try again in a moment.")
        else:
            prediction, audio_clip = process_audio(uploaded_file)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Result")
            if prediction[0][0] == 1:
                st.write("Prediction: Fake")
            else:
                st.write("Prediction: Real")

        with col2:
            st.header("Audio file")
            st.info("Your uploaded audio is below")
            st.audio(uploaded_file)

            fig = px.line(x=list(range(len(audio_clip))), y=audio_clip)
            fig.update_layout(
                title="Waveform plot",
                xaxis_title="Time",
                yaxis_title="Amplitude",
            )
            st.plotly_chart(fig)


if __name__ == "__main__":
    main()

#oo