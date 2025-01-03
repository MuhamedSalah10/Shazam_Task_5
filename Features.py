# File: audio_fingerprint.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, 
                            QHeaderView, QProgressBar, QPushButton, QFileDialog, QLabel)
import librosa
import numpy as np
from scipy.signal import find_peaks
import imagehash
from PIL import Image
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
class AudioFingerprint:
    def __init__(self):
        self.features = {}
        self.database_path = "fingerprints_db.json"
        self.load_features()


    def load_features(self):
        """Load precomputed fingerprints from the database file."""
        if os.path.exists(self.database_path):
            with open(self.database_path, 'r') as f:
                self.features = json.load(f)
                print("readed the database")
        # else:
        #     self.features = {}
        #     self.save_features()

    def save_features(self):
        """Save fingerprints to the database file."""
        with open(self.database_path, 'w') as f:
            json.dump(self.features, f)

    def precompute_fingerprints(self, database_folder):
        """Precompute and save fingerprints for all songs in the database."""
        songs = [f for f in os.listdir(database_folder) 
                 if f.lower().endswith(('.mp3', '.wav'))]
        for song in songs:
            song_path = os.path.join(database_folder, song)
            if song not in self.features:
                fingerprint = self.generate_fingerprint(song_path)
                if fingerprint:
                    self.features[song] = fingerprint
                    print(f"Fingerprint generated for {song}")
                else:
                    print(f"Failed to generate fingerprint for {song}")
        self.save_features()





        

#############################################################################################################3
    def extract_features(self, audio_data, sr):
        """Extract more robust features for audio fingerprinting"""
        features = {}
        
        # 1. Compute mel-spectrogram with more bands for better frequency resolution
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=sr,
            n_mels=128,
            fmax=8000,
            hop_length=512,
            n_fft=2048
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 2. Extract tempo (beats per minute)
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
        features['tempo'] = float(tempo)
        
        # 3. Extract MFCCs with more coefficients
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
        features['mfccs'] = mfccs.tolist()
        features['mfcc_deltas'] = librosa.feature.delta(mfccs).tolist()
        
        # 4. Extract pitch-related features
        chromagram = librosa.feature.chroma_cqt(y=audio_data, sr=sr)
        features['chroma'] = chromagram.tolist()
        
        # 5. Extract rhythm features (onset pattern)
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        features['onset_pattern'] = onset_env.tolist()
        
        # 6. Extract spectral features
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        features['spectral_contrast'] = spectral_contrast.tolist()
        
        # 7. Extract harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
        features['harmonic_ratio'] = float(np.mean(np.abs(y_harmonic)) / np.mean(np.abs(audio_data)))
        features['percussive_ratio'] = float(np.mean(np.abs(y_percussive)) / np.mean(np.abs(audio_data)))
        
        return features, mel_spec_db

    
    def compute_similarity(self, fingerprint1, fingerprint2):
        """Compute improved similarity measure between two fingerprints"""
        weights = {
            'mfccs': 0.3,
            'chroma': 0.2,
            'tempo': 0.1,
            'onset': 0.1,
            'spectral': 0.1,
            'harmonic': 0.1,
            'hash': 0.1
        }
        
        scores = []
        
        # 1. MFCC similarity
        mfcc_sim = np.mean([
            cosine_similarity(
                np.array(fingerprint1['features']['mfccs']).reshape(1, -1),
                np.array(fingerprint2['features']['mfccs']).reshape(1, -1)
            )[0][0],
            cosine_similarity(
                np.array(fingerprint1['features']['mfcc_deltas']).reshape(1, -1),
                np.array(fingerprint2['features']['mfcc_deltas']).reshape(1, -1)
            )[0][0]
        ])
        scores.append(('mfccs', mfcc_sim))
        
        # 2. Chroma similarity
        chroma_sim = cosine_similarity(
            np.array(fingerprint1['features']['chroma']).reshape(1, -1),
            np.array(fingerprint2['features']['chroma']).reshape(1, -1)
        )[0][0]
        scores.append(('chroma', chroma_sim))
        
        # 3. Tempo similarity
        tempo_sim = 1 - abs(
            fingerprint1['features']['tempo'] - fingerprint2['features']['tempo']
        ) / max(fingerprint1['features']['tempo'], fingerprint2['features']['tempo'])
        scores.append(('tempo', tempo_sim))
        
        # 4. Onset pattern similarity
        onset_sim = cosine_similarity(
            np.array(fingerprint1['features']['onset_pattern']).reshape(1, -1),
            np.array(fingerprint2['features']['onset_pattern']).reshape(1, -1)
        )[0][0]
        scores.append(('onset', onset_sim))
        
        # 5. Spectral contrast similarity
        spectral_sim = cosine_similarity(
            np.array(fingerprint1['features']['spectral_contrast']).reshape(1, -1),
            np.array(fingerprint2['features']['spectral_contrast']).reshape(1, -1)
        )[0][0]
        scores.append(('spectral', spectral_sim))
        
        # 6. Harmonic/Percussive similarity
        harmonic_sim = 1 - abs(
            fingerprint1['features']['harmonic_ratio'] - fingerprint2['features']['harmonic_ratio']
        )
        percussive_sim = 1 - abs(
            fingerprint1['features']['percussive_ratio'] - fingerprint2['features']['percussive_ratio']
        )
        scores.append(('harmonic', (harmonic_sim + percussive_sim) / 2))
        
        # 7. Hash similarity
        hash_sim = sum(h1 == h2 for h1, h2 in zip(
            fingerprint1['hashes'].values(), 
            fingerprint2['hashes'].values()
        )) / len(fingerprint1['hashes'])
        scores.append(('hash', hash_sim))
        
        # Compute weighted average
        final_similarity = sum(
            weights[name] * score for name, score in scores
        )
        
        return final_similarity
    def compute_perceptual_hash(self, mel_spec_db):
        """
        Compute perceptual hashes from the mel spectrogram.
        Returns multiple hashes computed from different regions of the spectrogram.
        """
        # Normalize the mel spectrogram to 0-255 range for image processing
        mel_spec_normalized = ((mel_spec_db - mel_spec_db.min()) * 255 / 
                            (mel_spec_db.max() - mel_spec_db.min())).astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(mel_spec_normalized)
        
        # Compute different types of perceptual hashes
        hashes = {
            'average_hash': str(imagehash.average_hash(img)),
            'phash': str(imagehash.phash(img)),
            'dhash': str(imagehash.dhash(img)),
            'whash': str(imagehash.whash(img))
        }
        
        # Compute hashes for different segments of the spectrogram
        width = mel_spec_normalized.shape[1]
        segment_size = width // 3
        
        for i in range(3):
            start = i * segment_size
            end = start + segment_size
            segment = Image.fromarray(mel_spec_normalized[:, start:end])
            hashes[f'segment_{i}_hash'] = str(imagehash.average_hash(segment))
        
        return hashes
        
    def generate_fingerprint(self, audio_path):
        """Generate a more comprehensive fingerprint."""
        try:
            # Load audio with a consistent duration
            audio_data, sr = librosa.load(audio_path, duration=30)  # Use first 30 seconds
            
            # Extract features
            features, mel_spec_db = self.extract_features(audio_data, sr)
            
            # Compute perceptual hashes
            hashes = self.compute_perceptual_hash(mel_spec_db)
            
            return {
                'name': os.path.basename(audio_path),
                'features': features,
                'hashes': hashes
            }
        except Exception as e:
            print(f"Error generating fingerprint for {audio_path}: {str(e)}")
            return None
