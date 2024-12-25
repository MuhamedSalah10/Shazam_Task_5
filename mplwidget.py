from PyQt5  import QtWidgets
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import librosa.display
from scipy.signal import spectrogram


class spec_Widget(QWidget):
    def __init__(self, parent=None):
        # create vertical layout >>put it the canvas and set this layout to the Qwidget
        super().__init__(parent)
        self.canvas = FigureCanvas(Figure(facecolor='none'))
        
        # self.setGeometry(830, 450, 1011, 371)
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.canvas.axes.axis('off')
        # # Set the color of the axes to white
        # self.canvas.axes.tick_params(axis='both', colors='white')
        self.setLayout(vertical_layout)


    def plot_spectrogram(self, data, sample_rate, x_label='Time', y_label='Frequency' ,  n_fft=2048):
        if data is None or len(data) == 0:
            return  # Exit the function if data is empty
        self.canvas.axes.clear()  # Clear previous plot
        n_fft = min(n_fft, len(data))
        try:
            spectrogram = librosa.amplitude_to_db(
                np.abs(librosa.stft(data, n_fft=n_fft)),
                ref=np.max
            )
        except ValueError as e:
            print("hereeee")
            return

        # Display the spectrogram with axes and labels
        librosa.display.specshow(
            spectrogram,
            sr=sample_rate,
            x_axis='time',
            y_axis='log',
            ax=self.canvas.axes,
            cmap='magma'  # Optional: set a color map
        )

        # Set labels if needed
        self.canvas.axes.set_xlabel(x_label)
        self.canvas.axes.set_ylabel(y_label)

        # Adjust padding to allow room for labels and ticks
        # self.canvas.figure.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.2)
        self.canvas.figure.subplots_adjust(left=0.1, right=1, top=1, bottom=0.08)
        self.canvas.draw()

        # pass
    


    # def plot_spec_parts(self , data , sample_rate, x_label='Time', y_label='Frequency'):
    #     if data is None or len(data) == 0:
    #         print("error")
    #         return  # Exit the function if data is empty
    #     # print("hello")
    #     # print(data.shape)
    #     # if data.ndim == 1:
    #     #     data = data.reshape(1025, 1)
    #     print(f"{len(data)}")
    #     self.canvas.axes.clear()
    #     if data.size > 0:
    #         librosa.display.specshow(
    #             data,
    #             sr=sample_rate,
    #             x_axis='time',
    #             y_axis='log',
    #             ax=self.canvas.axes,
    #             cmap='magma'  # Optional: set a color map
    #         )

    #     # Set labels if needed
    #     self.canvas.axes.set_xlabel(x_label)
    #     self.canvas.axes.set_ylabel(y_label)

    #     # Adjust padding to allow room for labels and ticks
    #     self.canvas.figure.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.2)
    #     self.canvas.draw()




    

    def clear(self):
        self.canvas.axes.clear()
        self.canvas.draw()
