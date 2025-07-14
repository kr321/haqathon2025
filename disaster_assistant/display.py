import sys
import cv2
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QTextEdit,
    QVBoxLayout, QHBoxLayout, QLineEdit, QFileDialog
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from app import photo

class DisasterChatUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Disaster Assistant")
        layout = QVBoxLayout()

        self.category_buttons = QHBoxLayout()
        for name in ["Earthquake", "Flood", "Fire", "Hurricane"]:
            btn = QPushButton(name)
            btn.clicked.connect(self.set_category)
            self.category_buttons.addWidget(btn)
        layout.addLayout(self.category_buttons)

        self.image_label = QLabel("No photo taken")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

        self.photo_button = QPushButton("Take Photo")
        self.photo_button.clicked.connect(self.take_photo)
        layout.addWidget(self.photo_button)

        input_row = QHBoxLayout()

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True) # This is j for display
        layout.addWidget(self.chat_display)

        self.user_input = QLineEdit()
        input_row.addWidget(self.user_input)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_row.addWidget(self.send_button)

        layout.addLayout(input_row)

        self.setLayout(layout)

        self.current_category = None

    def set_category(self):
        # should prob connect to logic ab the disaster cateogory
        button = self.sender()
        self.current_category = button.text()
        self.chat_display.append(f"Disaster Type: {self.current_category}")
        match self.current_category:
            case "Earthquake":
                self.chat_display.append(f"Expect aftershocks and avoid using open flames until utilities are cleared. Take cover if more shaking occurs.")
            case "Flood":
                self.chat_display.append(f"Move to the highest safe level of your home, but avoid attics without ventilation. Shut off electricity if water is rising. Don’t drink tap water unless declared safe. Keep emergency supplies dry.")
            case "Fire":
                self.chat_display.append(f"Close all windows and doors to prevent smoke from entering. Turn off ventilation systems, and stay low.")
            case "Hurricane":
                self.chat_display.append(f"Shelter in a windowless interior room on the lowest floor. Avoid using candles—opt for flashlights. Stay away from exterior walls and doors. If flooding begins, move to higher ground within your home.")
            case _:
                self.chat_display.append(f"")
        

    def take_photo(self):
        match self.current_category:
            case "Earthquake":
                out_text = photo(0)
            case "Flood":
                out_text = photo(1)
            case "Fire":
                out_text = photo(2)
            case "Hurricane":
                out_text = photo(3)
            case _:
                out_text = []
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            height, width, _ = frame.shape
            bytes_per_line = 3 * width
            qt_image = QImage(
                frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
            ).rgbSwapped()
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
            self.chat_display.append("Photo captured.")
        else:
            self.chat_display.append("Failed to take photo.")
        for i in out_text:
            self.chat_display.append(i)

    def send_message(self):
        msg = self.user_input.text()
        if msg:
            self.chat_display.append(f"You: {msg}")
            self.user_input.clear()
            # Add AI Response
            self.chat_display.append("AI: I’ll get you some advice based on that.")

app = QApplication(sys.argv)
window = DisasterChatUI()
window.show()
sys.exit(app.exec())
