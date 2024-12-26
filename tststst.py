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
        
    def extract_features(self, audio_data, sr):
        """Extract key audio features from the spectrogram"""
        features = {}
        
        # 1. Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, 
                                                n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 2. Extract spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            
        # 3. Extract spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        # 4. Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        features['mfccs_mean'] = np.mean(mfccs, axis=1).tolist()
        
        # 5. Extract chroma features
        chromagram = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        features['chroma_mean'] = np.mean(chromagram, axis=1).tolist()
        
        # 6. Find peaks in mel spectrogram
        peaks = []
        for i in range(mel_spec_db.shape[0]):
            peak_indices, _ = find_peaks(mel_spec_db[i, :])
            if len(peak_indices) > 0:
                peaks.extend([(int(i), int(j)) for j in peak_indices])
        features['peak_positions'] = peaks[:100]
        
        return features
    
    def compute_perceptual_hash(self, mel_spec_db):
        """Compute perceptual hash from mel spectrogram"""
        img_data = ((mel_spec_db - mel_spec_db.min()) * 255 / 
                   (mel_spec_db.max() - mel_spec_db.min())).astype(np.uint8)
        img = Image.fromarray(img_data)
        
        return {
            'average_hash': str(imagehash.average_hash(img)),
            'phash': str(imagehash.phash(img)),
            'dhash': str(imagehash.dhash(img)),
            'whash': str(imagehash.whash(img))
        }
    
    def generate_fingerprint(self, audio_path):
        """Generate complete fingerprint for an audio file"""
        try:
            audio_data, sr = librosa.load(audio_path)
            
            # Extract features
            features = self.extract_features(audio_data, sr)
            
            # Compute mel spectrogram for hashing
            mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Compute perceptual hashes
            hashes = self.compute_perceptual_hash(mel_spec_db)
            
            return {
                'features': features,
                'hashes': hashes
            }
        except Exception as e:
            print(f"Error generating fingerprint for {audio_path}: {str(e)}")
            return None
    
    def save_fingerprint(self, audio_path, fingerprint):
        """Save fingerprint to database"""
        try:
            with open(self.database_path, 'r') as f:
                database = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            database = {}
        
        database[os.path.basename(audio_path)] = fingerprint
        
        with open(self.database_path, 'w') as f:
            json.dump(database, f)
    
    def compute_similarity(self, fingerprint1, fingerprint2):
        """Compute similarity between two fingerprints"""
        try:
            # Compare MFCCs
            mfcc_sim = cosine_similarity(
                [fingerprint1['features']['mfccs_mean']], 
                [fingerprint2['features']['mfccs_mean']]
            )[0][0]
            
            # Compare chroma features
            chroma_sim = cosine_similarity(
                [fingerprint1['features']['chroma_mean']], 
                [fingerprint2['features']['chroma_mean']]
            )[0][0]
            
            # Compare spectral features
            centroid_diff = abs(fingerprint1['features']['spectral_centroid_mean'] - 
                              fingerprint2['features']['spectral_centroid_mean'])
            rolloff_diff = abs(fingerprint1['features']['spectral_rolloff_mean'] - 
                              fingerprint2['features']['spectral_rolloff_mean'])
            
            # Normalize differences
            max_centroid = max(fingerprint1['features']['spectral_centroid_mean'],
                             fingerprint2['features']['spectral_centroid_mean'])
            max_rolloff = max(fingerprint1['features']['spectral_rolloff_mean'],
                            fingerprint2['features']['spectral_rolloff_mean'])
            
            centroid_sim = 1 - (centroid_diff / max_centroid if max_centroid > 0 else 0)
            rolloff_sim = 1 - (rolloff_diff / max_rolloff if max_rolloff > 0 else 0)
            
            # Compare perceptual hashes
            hash_sim = sum(h1 == h2 for h1, h2 in zip(
                fingerprint1['hashes'].values(), 
                fingerprint2['hashes'].values()
            )) / len(fingerprint1['hashes'])
            
            # Weighted combination
            similarity_score = (0.3 * mfcc_sim + 
                              0.2 * chroma_sim + 
                              0.2 * centroid_sim +
                              0.1 * rolloff_sim +
                              0.2 * hash_sim)
            
            return max(0, min(1, similarity_score))  # Ensure score is between 0 and 1
            
        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            return 0

class SimilarityWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fingerprinter = AudioFingerprint()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Add buttons
        self.btn_load = QPushButton("Load Query Audio")
        self.btn_load.clicked.connect(self.load_query_audio)
        layout.addWidget(self.btn_load)
        
        self.btn_database = QPushButton("Select Database Directory")
        self.btn_database.clicked.connect(self.select_database_dir)
        layout.addWidget(self.btn_database)
        
        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Song', 'Similarity Score', 'Match %'])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        layout.addWidget(self.table)
        
        self.query_path = None
        self.database_dir = None
    
    def load_query_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Query Audio", "", "Audio Files (*.mp3 *.wav)")
        if file_path:
            self.query_path = file_path
            self.status_label.setText(f"Query loaded: {os.path.basename(file_path)}")
            if self.database_dir:
                self.find_similar_songs()
    
    def select_database_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Database Directory")
        if dir_path:
            self.database_dir = dir_path
            self.status_label.setText(f"Database directory: {dir_path}")
            if self.query_path:
                self.find_similar_songs()
    
    def find_similar_songs(self):
        """Find similar songs to the query audio"""
        if not self.query_path or not self.database_dir:
            return
        
        self.status_label.setText("Processing...")
        query_fingerprint = self.fingerprinter.generate_fingerprint(self.query_path)
        if not query_fingerprint:
            self.status_label.setText("Error processing query audio")
            return
        
        # Get list of songs in database
        songs = [f for f in os.listdir(self.database_dir) 
                if f.lower().endswith(('.mp3', '.wav'))]
        self.table.setRowCount(len(songs))
        self.progress.setMaximum(len(songs))
        
        similarities = []
        for i, song in enumerate(songs):
            song_path = os.path.join(self.database_dir, song)
            song_fingerprint = self.fingerprinter.generate_fingerprint(song_path)
            if song_fingerprint:
                similarity = self.fingerprinter.compute_similarity(
                    query_fingerprint, song_fingerprint)
                similarities.append((song, similarity))
            self.progress.setValue(i + 1)
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update table
        self.table.setRowCount(len(similarities))
        for i, (song, similarity) in enumerate(similarities):
            self.table.setItem(i, 0, QTableWidgetItem(song))
            self.table.setItem(i, 1, QTableWidgetItem(f"{similarity:.4f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{similarity * 100:.1f}%"))
        
        self.status_label.setText("Processing complete")