from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QSizePolicy, QApplication, 
                            QMainWindow, QRadioButton, QButtonGroup, QFileDialog, QTabWidget)
from PyQt5 import uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import librosa.display
import sys
import os
from scipy.io import wavfile
import logging
import sounddevice as sd
from PIL import Image, ImageQt, ImageEnhance
from numpy.fft import ifft2, ifftshift
from scipy.fft import fft2, fftshift
from mplwidget import spec_Widget
from PyQt5 import QtCore
from tststst import SimilarityWidget, AudioFingerprint
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl

# Configure logging
logging.basicConfig(
    filemode="a",
    filename="our_log.log",
    format="(%(asctime)s) | %(name)s| %(levelname)s | => %(message)s",
    level=logging.INFO
)

# Load the UI file
Ui_MainWindow, QtBaseClass = uic.loadUiType("First_UI.ui")

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        
        # Initialize variables
        self.isplay = False
        self.first_file = None
        self.second_file = None
        self.mixed_file = None
        self.played_sound = None
        self.paused_sound = None
        self.match_songs = [0]*8
        self.database_folder = "Weiner_Data"

        self.First_Song_Weight.sliderReleased.connect(lambda :self.mix_files(self.first_file, self.second_file))

        self.second_song_Weight.sliderReleased.connect(lambda :self.mix_files(self.first_file, self.second_file))
        
        # Connect buttons
        self.Upload_File_btn.clicked.connect(self.browse_file)
        self.play_signal_mixed.clicked.connect(lambda: self.play_sound('mixed'))
        self.play_signal_1.clicked.connect(lambda: self.play_sound('first'))
        self.play_signal_2.clicked.connect(lambda: self.play_sound('second'))
        
        # Connect output buttons
        for i in range(8):
            button_name = f"play_output_{i+1}"
            button = getattr(self, button_name)
            button.clicked.connect(lambda checked, idx=i: self.play_sound(f'output_{idx}'))
        
        # Initialize media player
        self.player = QMediaPlayer()
        self.player.stateChanged.connect(self.handle_state_changed)
        
        # Initialize timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        
        # Initialize spectrogram and fingerprint
        self.Spec_Org_obj = spec_Widget()
        self.setup_widget_layout(self.Spec_Org_obj, self.Spec_Org)
        self.fingerprinter = AudioFingerprint()
        self.query_path = None
    
    def setup_widget_layout(self, spec_widget, target_widget):
        """Setup the layout for spectrogram widgets"""
        if isinstance(target_widget, QWidget):
            layout = QVBoxLayout(target_widget)
            layout.addWidget(spec_widget)
            target_widget.setLayout(layout)
    
    def handle_state_changed(self, state):
        """Handle media player state changes"""
        if state == QMediaPlayer.StoppedState:
            self.played_sound = None
            self.paused_sound = None
        elif state == QMediaPlayer.PausedState:
            self.paused_sound = self.played_sound
            self.played_sound = None
    
    def browse_file(self):
        """Handle file browsing and loading"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Query Audio", "", "Audio Files (*.mp3 *.wav)")
        
        if not file_path:
            return
            
        if self.first_file is None:
            self.first_file = file_path
            curr_amplitude, curr_sample_rate = librosa.load(file_path, sr=None)
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.first_file)))
            self.Spec_Org_obj.plot_spectrogram(curr_amplitude, curr_sample_rate)
            self.find_similar_songs(file_path)
        else:
            self.player.stop()
            self.second_file = file_path
            self.mixed_file = self.mix_files(self.first_file, self.second_file)
            # curr_amplitude, curr_sample_rate = librosa.load(self.mixed_file, sr=None)
            # self.Spec_Org_obj.plot_spectrogram(curr_amplitude, curr_sample_rate)
            # self.Reset_prograssbars()
            # self.find_similar_songs(self.mixed_file)
    
    def play_sound(self, source):
        """Handle sound playback with proper pause/resume functionality"""
        if self.first_file is None and not source.startswith('output_'):
            return
            
        # Determine file path based on source
        file_path = None
        if source == 'mixed' and self.mixed_file:
            file_path = self.mixed_file
        elif source == 'first' and self.first_file:
            file_path = self.first_file
        elif source == 'second' and self.second_file:
            file_path = self.second_file
        elif source.startswith('output_'):
            idx = int(source.split('_')[1])
            if idx < len(self.match_songs):
                file_path = os.path.join(self.database_folder, self.match_songs[idx])
        
        if not file_path or not os.path.exists(file_path):
            print(f"Invalid file path: {file_path}")
            return
            
        # Handle play/pause logic
        if self.played_sound == source:
            # If the same source is currently playing, pause it
            print("Pausing current playback")
            self.player.pause()
            self.paused_sound = source
            self.played_sound = None
            
        elif self.paused_sound == source:
            # If this source was paused, resume it
            print("Resuming paused playback")
            self.player.play()
            self.played_sound = source
            self.paused_sound = None
            
        else:
            # If it's a new source, stop current playback and start new one
            print(f"Starting new playback: {file_path}")
            self.player.stop()
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
            self.player.play()
            self.played_sound = source
            self.paused_sound = None
            
        print(f"State after operation - Playing: {self.played_sound}, Paused: {self.paused_sound}")

    def handle_state_changed(self, state):
        """Handle media player state changes"""
        if state == QMediaPlayer.StoppedState:
            print("Player stopped")
            self.played_sound = None
            self.paused_sound = None
        elif state == QMediaPlayer.PausedState:
            print("Player paused")
        elif state == QMediaPlayer.PlayingState:
            print("Player playing")



    def find_similar_songs(self, path):
        """Find similar songs to the query audio"""
        if not path or not self.database_folder:
            return
        
        query_fingerprint = self.fingerprinter.generate_fingerprint(path)
        
        # Get list of songs in database
        songs = [f for f in os.listdir(self.database_folder) 
                if f.lower().endswith(('.mp3', '.wav'))]
        
        self.progress_calculations.setMaximum(len(songs))
        
        similarities = []
        for i, song in enumerate(songs):
            
            song_path = os.path.join(self.database_folder, song)
            song_fingerprint = self.fingerprinter.generate_fingerprint(song_path)
            if song_fingerprint:
                similarity = self.fingerprinter.compute_similarity(
                    query_fingerprint, song_fingerprint)
                similarities.append((song, similarity))
            self.progress_calculations.setValue(i + 1)
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update UI with results
        for i, (song, similarity) in enumerate(similarities[:8]):
            self.match_songs[i]=song
            
            self.groupBox_3.setTitle(f"Matching {os.path.splitext(os.path.basename(path))[0]}")
            progress_bar = getattr(self, f"progressBar_{i+1}", None)
            if progress_bar:
                progress_bar.setValue(int(similarity * 100))
            label = getattr(self, f"label_{i+1}", None)
            if label:
                label.setText(str(song))
        print(self.match_songs)
    
    def Reset_prograssbars(self):
        """Reset all progress bars and labels"""
        self.groupBox_3.setTitle(f"Matching")
        for i in range(8):
            progress_bar = getattr(self, f"progressBar_{i+1}", None)
            if progress_bar:
                progress_bar.setValue(0)
            label = getattr(self, f"label_{i+1}", None)
            if label:
                label.setText(f"Song_{i+1}")
    
    def mix_files(self, file1, file2):
        """Mix two audio files with weights from sliders"""
        # Read the two wav files

        if file1 is None or file2 is None:
            return 
        self.player.stop()
        rate1, data1 = wavfile.read(file1)
        rate2, data2 = wavfile.read(file2)
        
        # Ensure the sampling rates match
        if rate1 != rate2:
            rate1 = min(rate1, rate2)
        
        # Ensure the data lengths match by trimming
        min_length = min(len(data1), len(data2))
        data1 = data1[:min_length]
        data2 = data2[:min_length]
        
        # Normalize the data
        if np.issubdtype(data1.dtype, np.integer):
            data1 = data1 / np.iinfo(data1.dtype).max
        if np.issubdtype(data2.dtype, np.integer):
            data2 = data2 / np.iinfo(data2.dtype).max
        
        # Get weights from sliders
        weight1 = self.First_Song_Weight.value()
        weight2 = self.second_song_Weight.value()
        # total_weight = weight1 + weight2
        
        # if total_weight == 0:
        #     raise ValueError("Slider weights cannot both be zero")
        
        # Compute mixed audio
        mixed_data = ((weight1/100) * data1 + (weight2/100) * data2) 
        
        # Normalize mixed data
        mixed_data = mixed_data / np.max(np.abs(mixed_data))
        
        # Convert to 16-bit integer
        mixed_data = np.int16(mixed_data * 32767)
        
        # Save mixed file
        output_path = 'output_mix.wav'
        if os.path.exists(output_path):
            os.remove(output_path)
        wavfile.write(output_path, rate1, mixed_data)
        print("new mixxx")
        curr_amplitude, curr_sample_rate = librosa.load(output_path, sr=None)
        self.Spec_Org_obj.plot_spectrogram(curr_amplitude, curr_sample_rate)
        self.Reset_prograssbars()
        self.find_similar_songs(output_path)
        return output_path

if __name__ == "__main__":
    logging.info("----------------------the user open the app-------------------------------------")
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())