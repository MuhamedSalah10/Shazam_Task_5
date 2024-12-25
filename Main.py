from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QSizePolicy, QApplication, 
                            QMainWindow, QRadioButton, QButtonGroup, QFileDialog, QTabWidget)
from PyQt5 import uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Changed backend
from matplotlib.figure import Figure
import numpy as np
import librosa.display
import sys
import os
import logging
import sounddevice as sd
from PIL import Image, ImageQt, ImageEnhance
from numpy.fft import ifft2, ifftshift
from scipy.fft import fft2, fftshift
from mplwidget import spec_Widget
from PyQt5 import QtCore
from tststst import SimilarityWidget , AudioFingerprint
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
        

        self.Upload_File_btn.clicked.connect(self.browse_file)
        self.play_pause_btn.clicked.connect(self.play_pause)
        self.isplay=False
        self.curr_amplitude=None
        self.curr_sample_rate=0

        self.timer= QtCore.QTimer()
        self.timer.setInterval(50)
        self.frames=1024





        # Initialize spectrogram widgets
        self.Spec_Org_obj = spec_Widget()
        self.setup_widget_layout(self.Spec_Org_obj, self.Spec_Org)

        # # how deal with prograss barrr
        # self.progressBar_instrument_1.setValue(100)
        self.fingerprinter = AudioFingerprint()
        self.query_path = None
        self.database_dir = "Weiner_Data"
    


    def audio_callback(self, outdata, frames, time, status):
        """Callback function to send audio in chunks to the OutputStream."""
        try:
            # Determine the available samples
            remaining_samples = len(self.curr_amplitude) - self.frames
            if remaining_samples > 0:
                # Calculate the number of samples to send in this chunk
                samples_to_send = min(frames, remaining_samples)

                # Send the audio data for the current chunk
                outdata[:samples_to_send, 0] = self.curr_amplitude[self.frames:self.frames + samples_to_send]

                # Pad the rest with zeros if less than `frames` are available
                if samples_to_send < frames:
                    outdata[samples_to_send:, 0] = 0

                # Update the frame count in self.frames
                self.frames += samples_to_send
            else:
                # If no samples remain, fill output with zeros
                outdata.fill(0)

                # Stop the stream when no more data is available
                if self.stream.active:
                    self.stream.stop()

        except Exception as e:
            print(f"Error in audio_callback: {e}")


    
    def setup_widget_layout(self, spec_widget, target_widget):
        if isinstance(target_widget, QWidget):
            layout = QVBoxLayout(target_widget)
            layout.addWidget(spec_widget)
            target_widget.setLayout(layout) 
    
    # show Browser , return File path
    def browse_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;CSV Files (*.csv);;DAT Files (*.dat);;XLSX Files (*.xlsx);;TXT Files (*.txt)", options=options)
        
        if fileName.endswith('.wav'):
            self.curr_amplitude , self.curr_sample_rate = librosa.load(fileName , sr= None)
            self.curr_time = np.linspace(0 , len(self.curr_amplitude)/self.curr_sample_rate  ,self.curr_sample_rate )
            self.Spec_Org_obj.plot_spectrogram(self.curr_amplitude, self.curr_sample_rate )
            # Set up the stream (ensure self.curr_sample_rate and other relevant attributes are initialized)
            self.stream = sd.OutputStream(
                samplerate=self.curr_sample_rate,
                channels=1,
                callback=self.audio_callback)


            self.query_path = fileName
            if self.database_dir:
                self.groupBox_3.setTitle(f"Matching {fileName}")
                self.Reset_prograssbars()
                self.find_similar_songs()


    
    def find_similar_songs(self):
        """Find similar songs to the query audio"""
        if not self.query_path or not self.database_dir:
            return
        
        query_fingerprint = self.fingerprinter.generate_fingerprint(self.query_path)

        
        # Get list of songs in database
        songs = [f for f in os.listdir(self.database_dir) 
                if f.lower().endswith(('.mp3', '.wav'))]

        self.progress_calculations.setMaximum(len(songs))
        
        similarities = []
        for i, song in enumerate(songs):
            song_path = os.path.join(self.database_dir, song)
            song_fingerprint = self.fingerprinter.generate_fingerprint(song_path)
            if song_fingerprint:
                similarity = self.fingerprinter.compute_similarity(
                    query_fingerprint, song_fingerprint)
                similarities.append((song, similarity))
            self.progress_calculations.setValue(i + 1)
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(similarities)

        for i, (song, similarity) in enumerate(similarities):
            progress_bar = getattr(self,  f"progressBar_instrument_{i+1}" , None)
            progress_bar.setValue(int(similarity * 100))
            label=getattr(self,  f"label_{i+1}" , None)
            label.setText(str(song))
            if i==11:
                break
    
    def Reset_prograssbars(self):
        for i in range(11):
            progress_bar = getattr(self,  f"progressBar_instrument_{i+1}" , None)
            progress_bar.setValue(0)
    


            
    def play_pause(self):
        if self.isplay:
            self.isplay=False
            self.timer.stop()
            self.play_pause_btn.setText("play")
            if self.curr_amplitude is not None:
                self.stream.stop()
        else:
            self.isplay=True
            self.timer.start()
            self.play_pause_btn.setText("Pause")
            if self.curr_amplitude is not None:
                self.stream.start()

    # def reset(self):
    #     self.tracking_index=0
    #     self.timer.start()
    #     self.PushButton_PlayPause_Input.setText("Pause")
    #     self.isplay=True
    #     if self.mode.audio:
    #         self.stream.start()   

    

    
        

''
if __name__ == "__main__":
    logging.info("----------------------the user open the app-------------------------------------")
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())