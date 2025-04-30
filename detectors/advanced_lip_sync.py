import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf

class SyncNet(nn.Module):
    """
    A simplified SyncNet model that processes both face images and mel spectrograms.
    The model extracts embeddings from the visual and audio inputs; the cosine
    similarity between these embeddings serves as the lip sync confidence.
    """
    def __init__(self):
        super(SyncNet, self).__init__()
        # Visual branch: Input face image of shape (3, 112, 112)
        self.visual_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # (3,112,112) -> (64,112,112)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (64,56,56)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # -> (128,56,56)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (128,28,28)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # -> (256,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2)   # -> (256,14,14)
        )
        self.visual_fc = nn.Sequential(
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU()
        )
        # Audio branch: Input mel spectrogram of shape (1, 80, 16)
        self.audio_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # (1,80,16) -> (64,80,16)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (64,40,8)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # -> (128,40,8)
            nn.ReLU(),
            nn.MaxPool2d(2)   # -> (128,20,4)
        )
        self.audio_fc = nn.Sequential(
            nn.Linear(128 * 20 * 4, 256),
            nn.ReLU()
        )

    def forward(self, face, mel):
        # Process visual branch
        visual_feat = self.visual_conv(face)  # Expected shape: (B,256,14,14)
        visual_feat = visual_feat.view(visual_feat.size(0), -1)
        visual_embed = self.visual_fc(visual_feat)  # (B,256)
        visual_embed = F.normalize(visual_embed, p=2, dim=1)
        
        # Process audio branch
        audio_feat = self.audio_conv(mel)  # Expected shape: (B,128,20,4)
        audio_feat = audio_feat.view(audio_feat.size(0), -1)
        audio_embed = self.audio_fc(audio_feat)  # (B,256)
        audio_embed = F.normalize(audio_embed, p=2, dim=1)
        
        # Compute cosine similarity between visual and audio embeddings.
        similarity = (visual_embed * audio_embed).sum(dim=1)
        return similarity

class AdvancedLipSyncDetector:
    """
    Advanced Lip Sync Detector using a deep-learning model (SyncNet).
    
    This module calculates a lip sync confidence value given a face region (ROI) and a corresponding
    audio clip. The steps include:
      - Preprocessing the face ROI: resizing to 112x112, normalizing pixels.
      - Preprocessing the audio clip: computing a mel spectrogram with 80 mel bins and fixed 16 time frames.
      - Passing the preprocessed inputs through the SyncNet model to extract embeddings.
      - Computing cosine similarity between embeddings as the lip sync confidence.
    """
    def __init__(self, model_path: str = "models/Wav2Lip-SD-NOGAN.pt", sample_rate: int = 16000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        # Initialize the SyncNet model.
        self.model = SyncNet().to(self.device)
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"[AdvancedLipSyncDetector] Loaded model from {model_path}")
        else:
            print(f"[AdvancedLipSyncDetector] Model path {model_path} is invalid. Using untrained model.")
        self.model.eval()

    def preprocess_face(self, face_roi: np.ndarray) -> torch.Tensor:
        """
        Preprocess the face image:
         - Resize to 112x112.
         - Normalize pixel values to [0, 1].
         - Convert to a PyTorch tensor with shape (1, 3, 112, 112).
        """
        face_resized = cv2.resize(face_roi, (112, 112))
        face_normalized = face_resized.astype(np.float32) / 255.0  
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        return face_tensor

    def preprocess_audio(self, audio_clip: np.ndarray) -> torch.Tensor:
        """
        Preprocess the audio clip:
         - Compute a mel spectrogram using librosa with 80 mel bins.
         - Convert the power spectrogram to decibel scale.
         - Normalize the spectrogram to [0, 1].
         - Fix the time dimension to 16 frames (via padding/truncation).
         - Convert to a PyTorch tensor with shape (1, 1, 80, 16).
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio_clip,
            sr=self.sample_rate,
            n_mels=80,
            n_fft=400,
            hop_length=160
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec_norm = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-6)
        # Ensure a fixed time dimension of 16 frames.
        if log_mel_spec_norm.shape[1] < 16:
            pad_width = 16 - log_mel_spec_norm.shape[1]
            log_mel_spec_norm = np.pad(log_mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel_spec_norm = log_mel_spec_norm[:, :16]
        mel_tensor = torch.from_numpy(log_mel_spec_norm).unsqueeze(0).unsqueeze(0).float()
        mel_tensor = mel_tensor.to(self.device)
        return mel_tensor

    def calculate_lip_sync_value(self, face_roi: np.ndarray, audio_clip: np.ndarray) -> float:
        """
        Calculate the lip sync confidence given a face ROI and its corresponding audio clip.
        
        :param face_roi: Face region (color image) as a NumPy array.
        :param audio_clip: Audio clip (raw waveform, mono) as a 1D NumPy array.
        :return: Lip sync confidence value (cosine similarity) as a float.
        """
        face_tensor = self.preprocess_face(face_roi)
        mel_tensor = self.preprocess_audio(audio_clip)
        with torch.no_grad():
            similarity = self.model(face_tensor, mel_tensor)
        lip_sync_value = similarity.item()
        return lip_sync_value

if __name__ == "__main__":
    # Test block to verify functionality.
    face_path = "test_face.jpg"
    audio_path = "test_audio.wav"

    if not os.path.exists(face_path):
        raise FileNotFoundError(f"Test face image not found: {face_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Test audio file not found: {audio_path}")

    # Load face image and convert from BGR to RGB.
    face_image = cv2.imread(face_path)
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Load audio using soundfile.
    audio_data, sr = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    target_sr = 16000
    if sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)

    # Initialize the advanced lip sync detector using the downloaded model.
    detector = AdvancedLipSyncDetector(model_path="models/Wav2Lip-SD-NOGAN.pt", sample_rate=target_sr)
    lip_sync_confidence = detector.calculate_lip_sync_value(face_image, audio_data)
    print("Lip Sync Confidence Value:", lip_sync_confidence)