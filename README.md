# DeepFakeAudio
Deepfake content is created or altered synthetically using artificial intelligence (AI) approaches to appear real. It can include synthesizing audio, video, images, and text. Deepfakes may now produce natural-looking content, making them harder to identify. Deepfake audio technology has several disadvantages, including the potential for spreading misinformation and fake news, privacy concerns, ethical concerns, and security risks. It can create convincing fake recordings of public figures, politicians, or celebrities, posing a threat to public trust and society. Privacy concerns arise from creating recordings without consent, potentially leading to malicious activities like blackmail or harassment. Ethical concerns arise from the difficulty in determining the authenticity of recordings, especially in the increasingly sophisticated technology. Security risks arise from the potential for unauthorized access to sensitive information or systems, particularly in sectors like finance, healthcare, and law enforcement. Much progress has been achieved in identifying video deepfakes in recent years; nevertheless, most investigations in detecting audio deepfakes have employed the ASVSpoof or AVSpoof dataset and various machine learning, deep learning, and deep learning algorithms. This research uses machine and deep learning-based approaches such as MFCCs, neural networks etc., to identify deepfake audio. This project focuses on building a deep learning model for classifying audio files as either genuine (bonafide) or manipulated (spoof). The objective is to detect audio deepfakes, which are manipulated audio recordings designed to impersonate a genuine audio source. The ASVspoof 2019 dataset is used for training and evaluating the model.

## Hugging Face Deployment

This app is designed for a private Hugging Face Space using Streamlit. The model can be stored in a private Hugging Face model repository and loaded at app startup.

### 1. Create a private model repository

1. Create a Hugging Face account.
2. Go to New Model.
3. Use a name like `deepfake-audio-classifier`.
4. Set visibility to Private.
5. Upload `classifier.h5` to the model repository.

For a 74 MB model, use Git LFS or upload from the Hugging Face web UI.

### 2. Create a private Streamlit Space

1. Go to New Space.
2. Use a name like `deepfake-audio-detector`.
3. Select `Streamlit` as the SDK.
4. Set visibility to Private.
5. Upload this project to the Space repository:
   - `app.py`
   - `requirements.txt`
   - `README.md`

### 3. Add Space secrets

In the Space settings, add these secrets:

```text
HF_MODEL_REPO_ID=your-username/deepfake-audio-classifier
HF_MODEL_FILENAME=classifier.h5
HF_TOKEN=your_hugging_face_access_token
```

Create the Hugging Face token from Settings > Access Tokens. It needs read access to the private model repository.

### 4. Run the Space

After the files and secrets are added, restart the Space. The app should show two tabs:

- Record audio from the browser microphone.
- Upload `.flac`, `.wav`, or `.mp3` audio.

The model returns `Fake` when the prediction output is exactly `1`; otherwise it returns `Real`.
