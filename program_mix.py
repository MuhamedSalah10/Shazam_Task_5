import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog
import scipy.signal
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import numpy as np, bisect, librosa, soundfile as sf
import os
import numpy as np
from scipy.io import wavfile
import tempfile
import os
from PyQt5.QtCore import QTimer
from tststst import AudioFingerprint 
# Load ui
Ui_MainWindow, QtBaseClass = uic.loadUiType("ui5.ui")

temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        
        self.open_button.clicked.connect(self.open_file)
        self.open_folder.clicked.connect(self.select_folder)
        self.current_file = None
        self.first_file = self.second_file = None
        self.database_folder = None
        
        self.match_songs = [None, None, None, None, None]
        
        self.songs_labels = [self.first_song_label, self.second_song_label, self.third_song_label, self.fourth_song_label, self.fifth_song_label]
        self.songs_progress_bars = [self.first_progressBar, self.second_progressBar, self.third_progressBar, self.fourth_progressBar, self.fifth_progressBar]
       
        self.sound_button.clicked.connect(lambda: self.play_sound("loaded_file"))
        self.input1_sound.clicked.connect(lambda: self.play_sound("input1_sound"))
        self.input1_sound_2.clicked.connect(lambda: self.play_sound("input2_sound"))
        
        self.output_sound_1.clicked.connect(lambda: self.play_sound(0))
        self.output_sound_2.clicked.connect(lambda: self.play_sound(1))
        self.output_sound_3.clicked.connect(lambda: self.play_sound(2))
        self.output_sound_4.clicked.connect(lambda: self.play_sound(3))
        self.output_sound_5.clicked.connect(lambda: self.play_sound(4))
        
        self.player = QMediaPlayer()
        self.played_sound = None
        self.paused_sound = None
        
        self.fingerprinter = AudioFingerprint()
        
    
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Query Audio", "", "Audio Files (*.mp3 *.wav)")
        
        print(file_path)
        if self.current_file:
            self.second_file = file_path
            self.mix_files(self.first_file, self.second_file)
            self.mix_button.clicked.connect(lambda: self.mix_files(self.first_file, self.second_file))
            if self.database_folder:
                self.find_similar_songs()
            
        elif file_path:
            self.current_file = file_path
            self.first_file = self.current_file
            if self.database_folder:
                self.find_similar_songs()
    
    def select_folder(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Database Directory")
        if dir_path:
            self.database_folder = dir_path
            if self.current_file:
                self.find_similar_songs() 
                
                
    def find_similar_songs(self):
        """Find similar songs to the query audio"""
        if not self.current_file or not self.database_folder:
            return
        
        query_fingerprint = self.fingerprinter.generate_fingerprint(self.current_file)
        if not query_fingerprint:
            return
        
        # Get list of songs in database
        songs = [f for f in os.listdir(self.database_folder) 
                if f.lower().endswith(('.mp3', '.wav'))]
        
        similarities = []
        for i, song in enumerate(songs):
            song_path = os.path.join(self.database_folder, song)
            song_fingerprint = self.fingerprinter.generate_fingerprint(song_path)
            if song_fingerprint:
                similarity = self.fingerprinter.compute_similarity(
                    query_fingerprint, song_fingerprint)
                similarities.append((song, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        print("similarities: ", similarities)
        
        for i, (song, similarity) in enumerate(similarities):
            self.match_songs[i] = song
            print("i = ", i, "song: ", song, "similarity: ", similarity)
            self.songs_labels[i].setText(song) 
            self.songs_progress_bars[i].setValue(int(similarity * 100))
            if i == 4:
                break
            
    def mix_files(self, file1, file2):
        # Read the two wav files
        rate1, data1 = wavfile.read(file1)
        rate2, data2 = wavfile.read(file2)

        # Ensure the sampling rates match
        if rate1 != rate2:
            rate1 = min(rate1, rate2)
            # raise ValueError("Sampling rates of the two .wav files must match")

        # Ensure the data lengths match by trimming or padding
        min_length = min(len(data1), len(data2))
        data1 = data1[:min_length]
        data2 = data2[:min_length]

        # Normalize the data based on type
        if np.issubdtype(data1.dtype, np.integer):
            data1 = data1 / np.iinfo(data1.dtype).max
        if np.issubdtype(data2.dtype, np.integer):
            data2 = data2 / np.iinfo(data2.dtype).max

        # Get weights from sliders
        weight1 = self.input1_slider.value()
        weight2 = self.input2_slider.value()
        total_weight = weight1 + weight2
        if total_weight == 0:
            raise ValueError("Slider weights cannot both be zero")

        # Compute mixed audio
        mixed_data = (weight1 * data1 + weight2 * data2) / total_weight

        # Normalize mixed data to fit within [-1, 1]
        mixed_data = mixed_data / np.max(np.abs(mixed_data))

        # Convert back to the original data type
        if np.issubdtype(data1.dtype, np.integer):
            mixed_data = (mixed_data * np.iinfo(data1.dtype).max).astype(data1.dtype)
        else:
            # If floating-point, ensure it's in the same dtype as original
            mixed_data = mixed_data.astype(data1.dtype)

        # Save the mixed data to a new .wav file
        mixed_data = np.int16(mixed_data / np.max(np.abs(mixed_data)) * 32767)
        if os.path.exists('output_mix.wav'):
            os.remove('output_mix.wav')
        wavfile.write('output_mix.wav', rate1, mixed_data)
        self.current_file = 'output_mix.wav'

        # Check mixed data range before saving
        print("Mixed data max:", np.max(np.abs(mixed_data)))
        print("Mixed data min:", np.min(np.abs(mixed_data)))

        if self.played_sound == "loaded_file":
            self.played_sound = None
            self.paused_sound = None
            self.player.stop()
            self.player = QMediaPlayer()
            self.play_sound("loaded_file")
        

    def play_sound(self, button):
        if self.current_file is None:
            return
        
        if button == self.played_sound:
            print("pause")
            self.played_sound = None
            self.paused_sound = button
            self.player.pause()
            return
        
        elif self.paused_sound == button:
            print("play")
            self.played_sound = button
            self.player.play()
            return
        
        if button == "loaded_file": 
            self.player.stop()   
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.current_file)))
            self.player.play()
            
        elif button == "input1_sound":
            self.player.stop()
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.first_file)))
            self.player.play()    
            
        elif button == "input2_sound":
            self.player.stop()
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.second_file)))
            self.player.play()     
            
        elif self.match_songs[button]:
            output_path = os.path.join(self.database_folder, self.match_songs[button])
            print("self.match_songs[button]: ", output_path)
            self.player.stop()
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(output_path)))
            self.player.play()   
            
        self.played_sound = button    
        self.paused_sound = None
        print("self.played_sound: ", self.played_sound ,"self.paused_sound: ", self.paused_sound )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())