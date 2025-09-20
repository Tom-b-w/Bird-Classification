from transformers import AutoModelForSequenceClassification, AutoFeatureExtractor, Trainer, TrainingArguments
import torch
import librosa

# Load the feature extractor and the base model
model_name = "DBD-research-group/Bird-MAE-Base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True,
                                                           num_labels=3)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)

# Set the model to evaluation mode
model.eval()


def classify_bird(audio_path):
    # Load an example audio file with the required sampling rate of 32kHz
    audio, sample_rate = librosa.load(audio_path, sr=32_000)

    # Extract the Mel spectrogram from the audio
    mel_spectrogram = feature_extractor(audio, return_tensors="pt")

    # Predict with the model
    with torch.no_grad():
        outputs = model(**mel_spectrogram)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()

    return predicted_class_id

# Example usage: classify a bird from an audio file
audio_path = "D:/Bird-Sound-Recogniser-main/Bird-Sound-Recogniser-main/static\mixed_audio/大天鹅_大天鹅_with_灰雁.mp3" # Replace with your audio file path
bird_class = classify_bird(audio_path)

print(f"Predicted bird class ID: {bird_class}")
