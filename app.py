import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import plotly.express as px

# Set page config
st.set_page_config(page_title="Speech Emotion Recognition", layout="wide")

# Load model and processor from local directory
def load_local_model():
    model_path = "C:\\Users\\dhruv\\My_model"  # Local path to the downloaded model
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    return model, processor

# Load model and processor once
if 'model' not in st.session_state:
    st.session_state.model, st.session_state.processor = load_local_model()

# Function to extract features and make predictions
def predict_emotion(audio_data):
    # Preprocess audio data
    inputs = st.session_state.processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = st.session_state.model(**inputs).logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return predicted_class_id

# Function to plot waveform
def plot_waveform(audio_data, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sr, ax=ax)
    plt.title('Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    return fig

# Function to plot spectrogram
def plot_spectrogram(audio_data, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.title('Spectrogram')
    return fig

# Define emotion labels and corresponding emojis
emotions = {
    'angry': '😠',
    'calm': '😌',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}

# Main app function
def main():
    st.title("Speech Emotion Recognition")
    
    # Sidebar for uploading audio
    st.sidebar.title("Upload Audio File")
    uploaded_file = st.sidebar.file_uploader("Upload a .wav file", type=['wav'])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # Load audio file
        audio_data, sample_rate = librosa.load(uploaded_file, sr=16000)
        
        # Show visualizations
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Waveform")
            st.pyplot(plot_waveform(audio_data, sample_rate))
        with col2:
            st.subheader("Spectrogram")
            st.pyplot(plot_spectrogram(audio_data, sample_rate))
        
        # Predict emotion
        predicted_class_id = predict_emotion(audio_data)
        predicted_emotion = list(emotions.keys())[predicted_class_id]

        # Display prediction
        st.subheader("Prediction Result")
        st.write(f"### Predicted Emotion: **{predicted_emotion.upper()}** {emotions[predicted_emotion]}")

        # Display confidence scores as a bar chart
        st.subheader("Confidence Scores")
        with torch.no_grad():
            inputs = st.session_state.processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
            logits = st.session_state.model(**inputs).logits
            confidence_scores = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]

        # Calculate percentages
        confidence_percentages = confidence_scores * 100

        # Create a DataFrame for display
        emotion_labels = list(emotions.keys())
        confidence_data = {
            'Emotion': [f"{emotions[emotion]} {emotion.capitalize() }" for emotion in emotion_labels],
            'Confidence (%)': confidence_percentages
        }

        # Create a bar chart with percentages
        fig = px.bar(confidence_data, x='Emotion', y='Confidence (%)', labels={'x': 'Emotion', 'y': 'Confidence (%)'})
        st.plotly_chart(fig)

    else:
        st.info("Please upload an audio file to start.")

    # Add information about the app
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This app performs speech emotion recognition on uploaded audio files.
    
    Supported emotions:
    - Angry
    - Calm
    - Disgust
    - Fear
    - Happy
    - Neutral
    - Sad
    """)

if __name__ == "__main__":
    main()
