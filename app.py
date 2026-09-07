import os
import tempfile
import logging
import hashlib

import librosa
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
HF_MODEL_REPO_ID = os.environ.get("HF_MODEL_REPO_ID", "")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "classifier.h5")


st.set_page_config(
    page_title="DeepFake Audio Detector",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource
def load_model():
    if HF_MODEL_REPO_ID and not os.path.exists(MODEL_PATH):
        try:
            from huggingface_hub import hf_hub_download

            downloaded_model_path = hf_hub_download(
                repo_id=HF_MODEL_REPO_ID,
                filename=HF_MODEL_FILENAME,
                token=os.environ.get("HF_TOKEN"),
            )
            logging.info(
                "Model downloaded from Hugging Face: repo=%s filename=%s path=%s",
                HF_MODEL_REPO_ID,
                HF_MODEL_FILENAME,
                downloaded_model_path,
            )
            return keras.models.load_model(downloaded_model_path)
        except Exception as exc:  # pragma: no cover - deployment-time diagnostics
            raise RuntimeError(
                f"Failed to obtain model from Hugging Face repo '{HF_MODEL_REPO_ID}': {exc}"
            ) from exc

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


def process_audio(audio_source, suffix=".flac"):
    temp_file_path = None

    try:
        if isinstance(audio_source, bytes):
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(audio_source)
                temp_file_path = temp_file.name
        elif hasattr(audio_source, "read"):
            audio_source.seek(0)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
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


def audio_fingerprint(audio_bytes):
    return hashlib.sha256(audio_bytes).hexdigest()[:12]


def format_prediction(prediction):
    score = float(prediction[0][0])
    label = "Fake" if prediction[0][0] == 1 else "Real"
    return label, score


def render_result(prediction, audio_clip, audio_source):
    label, score = format_prediction(prediction)
    status = "error" if label == "Fake" else "success"

    result_col, audio_col = st.columns([0.95, 1.35], gap="large")

    with result_col:
        st.subheader("Result")
        getattr(st, status)(f"Prediction: {label}")
        st.metric("Model score", f"{score:.4f}")

        st.warning(
            "This detector should be treated as a decision-support signal, not final proof."
        )

    with audio_col:
        st.subheader("Audio")
        st.audio(audio_source)

        fig = px.line(x=list(range(len(audio_clip))), y=audio_clip)
        fig.update_layout(
            title="Waveform",
            xaxis_title="Sample",
            yaxis_title="Amplitude",
            height=320,
            margin=dict(l=24, r=24, t=48, b=24),
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    try:
        load_model()
    except Exception as exc:
        st.error(f"Unable to load the model. Error: {exc}")
        return

    st.title("DeepFake Audio Detector")
    st.caption("Record or upload a short audio clip to classify it as real or fake.")
    st.divider()

    record_tab, upload_tab = st.tabs(["Record", "Upload"])

    with record_tab:
        st.subheader("Record Audio")
        audio = mic_recorder(start_prompt="Record", stop_prompt="Stop", key="recorder")

        if audio:
            audio_bytes = audio["bytes"]
            st.caption(f"Audio ID: {audio_fingerprint(audio_bytes)}")
            prediction, audio_clip = process_audio(audio_bytes, suffix=".flac")
            render_result(prediction, audio_clip, audio_bytes)

    with upload_tab:
        st.subheader("Upload Audio")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["flac", "wav", "mp3"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            suffix = os.path.splitext(uploaded_file.name)[1] or ".flac"
            st.caption(f"Audio ID: {audio_fingerprint(uploaded_bytes)}")
            prediction, audio_clip = process_audio(uploaded_bytes, suffix=suffix)
            render_result(prediction, audio_clip, uploaded_bytes)


if __name__ == "__main__":
    main()
