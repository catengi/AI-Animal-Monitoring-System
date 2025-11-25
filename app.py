import cv2
import numpy as np
import sys
import os
import json 
import threading # เพิ่ม: สำหรับรัน Flask ใน Thread แยก
import time

from flask import Flask, jsonify
from werkzeug.serving import run_simple # ใช้ run_simple เพื่อการจัดการ Thread ที่ดีกว่า app.run ปกติ

from ultralytics import YOLO

# Import necessary PyQt modules (using PyQt6)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QSlider, QRadioButton
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
from PyQt6.QtCore import (
    QThread, pyqtSignal, Qt
)

# --- 1. CONFIGURATION AND INITIALIZATION ---

MODEL_PATH = "bison_s_100ep.pt"
CONFIG_FILE = "region_config.json" # File to store region settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.75 
ALERT_TEXT = "!!! REGION BREACH !!!"
API_PORT = 5000 # เพิ่ม: พอร์ตสำหรับ API

# --- 2. CORE GEOMETRY LOGIC ---

def is_point_on_line_side(P, P1_line, P2_line):
    """
    Determines which side of the line segment P1_line -> P2_line the point P is on.
    Returns 1 (one side), -1 (other side), or 0 (on line).
    """
    # Cross product (P2_line - P1_line) x (P - P1_line)
    val = (P2_line[0] - P1_line[0]) * (P[1] - P1_line[1]) - \
          (P2_line[1] - P1_line[1]) * (P[0] - P1_line[0])
    
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0

def check_line_crossing(box, P1_line, P2_line):
    """
    Checks if the bounding box intersects the line segment P1_line -> P2_line.
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Define the four corners of the bounding box
    corners = [
        (x1, y1), (x2, y1), 
        (x1, y2), (x2, y2)
    ]

    initial_side = is_point_on_line_side(corners[0], P1_line, P2_line)
    
    # If all points are the same (a dot box), avoid division by zero edge case
    if initial_side == 0 and all(c == corners[0] for c in corners):
        return False
        
    for i in range(1, 4):
        current_side = is_point_on_line_side(corners[i], P1_line, P2_line)
        
        # Crossing is detected if corners are on opposite sides (initial_side * current_side < 0)
        # or if one corner is on the line (current_side == 0).
        if current_side != 0 and initial_side != 0 and current_side != initial_side:
            return True
            
    return False

def is_box_inside_polygon(box, polygon_points):
    """
    Checks if the center of the bounding box is inside the polygon defined by polygon_points.
    polygon_points should be a list of (x, y) tuples.
    Returns True if inside or on the boundary.
    """
    if len(polygon_points) < 3:
        return False
        
    x1, y1, x2, y2 = map(int, box)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    center_point = (center_x, center_y)

    # Convert the list of tuples to a numpy array for cv2 function
    pts = np.array(polygon_points, np.int32)
    
    # Reshape to (N, 1, 2) to represent a contour for cv2
    pts = pts.reshape((-1, 1, 2))
    
    # Check if the point is inside (returns positive value if inside, 0 on edge, negative if outside)
    result = cv2.pointPolygonTest(pts, center_point, measureDist=False)
    
    # Result >= 0 means inside or on the boundary
    return result >= 0

# --- 3. VIDEO WORKER THREAD ---

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    breach_signal = pyqtSignal(bool)
    
    # เพิ่ม: Signal สำหรับส่งข้อมูลดิบ (JSON-ready list) กลับไปที่ Main Thread เพื่อให้ API ใช้
    detection_data_signal = pyqtSignal(list)
    
    P1 = None
    P2 = None
    confidence = DEFAULT_CONFIDENCE_THRESHOLD
    
    # New attributes for polygon mode
    region_mode = 'LINE' # 'LINE' or 'POLYGON'
    polygon_points = [] # List of polygon vertices
    temp_P = None # Temporary point for polygon drawing preview

    
    def __init__(self, cap_source, parent=None):
        super().__init__(parent)
        self.cap_source = cap_source
        self.run_flag = True
        self.model = None
        self.cap = None

    def run(self):
        try:
            self.model = YOLO(MODEL_PATH)
            print("YOLO model loaded in worker thread.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.run_flag = False

        self.cap = cv2.VideoCapture(self.cap_source)
        if not self.cap.isOpened():
            print(f"Error: Could not open video source: {self.cap_source}")
            self.run_flag = False

        while self.run_flag and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # แก้ไข: รับค่า detection_list เพิ่มเติมจากฟังก์ชัน process_frame
                frame, detection_list, is_breach = self.process_frame(frame)
                
                self.change_pixmap_signal.emit(frame)
                self.breach_signal.emit(is_breach)
                
                # เพิ่ม: ส่งข้อมูล detection ไปอัปเดตตัวแปรสำหรับ API
                self.detection_data_signal.emit(detection_list)
            else:
                if self.cap_source != 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    print("Camera disconnected or error reading frame.")
                    break
            
            QThread.msleep(30) 

        if self.cap:
            self.cap.release()
        print("Video thread finished.")

    def process_frame(self, frame):
        """Processes the frame with YOLO and draws annotations."""
        is_breach = False
        detection_list = [] # เพิ่ม: เก็บข้อมูลวัตถุที่เจอในเฟรมนี้
        
        # Use the dynamic confidence value from the thread's attribute
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        
        # 1. Check for Breach
        for result in results:
            boxes_data = result.boxes.data.cpu().numpy()
            
            for box_data in boxes_data:
                x1, y1, x2, y2, conf, cls = box_data
                box = [x1, y1, x2, y2]
                
                is_box_breaching = False
                
                # --- Check for Region Breach based on Mode ---
                if self.region_mode == 'LINE' and self.P1 and self.P2:
                    if check_line_crossing(box, self.P1, self.P2):
                        is_box_breaching = True
                        
                elif self.region_mode == 'POLYGON' and len(self.polygon_points) >= 3:
                    # Centroid check is suitable for Polygon mode
                    if is_box_inside_polygon(box, self.polygon_points):
                        is_box_breaching = True

                if is_box_breaching:
                    is_breach = True
                
                # เพิ่ม: เก็บข้อมูลลง List (ต้องแปลงเป็น float/int ปกติเพื่อให้ JSON ใช้งานได้)
                detection_list.append({
                    "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class_id": int(cls),
                    "is_breaching": is_box_breaching
                })

                # --- Draw Bounding Box ---
                box_color = (0, 0, 255) if is_box_breaching else (0, 255, 0) # Red or Green (BGR)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                
                # --- Calculate and Draw Centroid ---
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Draw a small, filled circle (dot) at the center (Cyan/Yellow BGR)
                centroid_color = (255, 255, 0) 
                cv2.circle(frame, (center_x, center_y), 5, centroid_color, -1) 
                # ----------------------------------------
                
                # Draw label 
                label = f"Bison: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # 2. Draw Control Region and Alert
        line_color = (0, 255, 255) # Default Yellow
        if is_breach:
            line_color = (0, 0, 255) # Red for breach
        
        if self.region_mode == 'LINE' and self.P1 and self.P2:
            # Draw line mode
            cv2.line(frame, self.P1, self.P2, line_color, 3)
            cv2.circle(frame, self.P1, 5, line_color, -1)
            cv2.circle(frame, self.P2, 5, line_color, -1)
            
        elif self.region_mode == 'POLYGON' and self.polygon_points:
            # Draw finalized polygon (if >= 3 points)
            pts = np.array(self.polygon_points, np.int32)
            if len(pts) >= 3:
                pts = pts.reshape((-1, 1, 2))
                # Draw the filled polygon with some transparency
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], line_color)
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                # Draw the polygon outline
                cv2.polylines(frame, [pts], isClosed=True, color=line_color, thickness=3)

            # Draw temporary line to cursor during drawing
            if len(self.polygon_points) >= 1 and self.temp_P:
                last_point = self.polygon_points[-1]
                cv2.line(frame, last_point, self.temp_P, (255, 255, 255), 2) # White preview line
                
            # Draw vertices
            for point in self.polygon_points:
                cv2.circle(frame, point, 5, line_color, -1)
        
        # return frame, detection_list, is_breach # เพิ่ม: Return detection_list
        return frame, detection_list, is_breach

    def stop(self):
        """Stops the thread gracefully."""
        self.run_flag = False
        self.wait()

# --- 4. FLASK API THREAD (ส่วนใหม่: สำหรับสร้าง API Server) ---

class ApiThread(threading.Thread):
    def __init__(self, app, host='0.0.0.0', port=API_PORT):
        super().__init__()
        self.app = app
        self.host = host
        self.port = port
        self.daemon = True # ปิด thread นี้อัตโนมัติเมื่อ Main Program ปิด

    def run(self):
        print(f"\nAPI Server started at http://localhost:{self.port}/status")
        # ใช้ run_simple เพื่อเลี่ยงปัญหา Flask reloader ใน Thread
        run_simple(self.host, self.port, self.app, threaded=True)

# --- 5. MAIN PYQT APPLICATION WINDOW ---

class MonitorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Animal Monitoring System (with API)")
        
        # Set large initial/minimum size
        self.setMinimumSize(1000, 750) 
        
        # State variables
        self.video_thread = None
        self.P1 = None
        self.P2 = None
        self.is_drawing = False
        self.video_running = False
        # Initialize confidence state variable
        self.confidence = DEFAULT_CONFIDENCE_THRESHOLD 
        
        # New state variables for region mode
        self.region_mode = 'LINE' # 'LINE' or 'POLYGON'
        self.polygon_points = [] # List of polygon vertices
        self.temp_P = None # Temporary point for line drawing while building polygon

        # เพิ่ม: ตัวแปร Shared State สำหรับ API
        self.api_data = {
            "status": "idle",
            "is_breach": False,
            "detections": [],
            "timestamp": 0,
            "mode": self.region_mode
        }
        self.data_lock = threading.Lock() # Lock เพื่อป้องกันข้อมูลชนกันระหว่าง Thread

        self.init_flask_api() # เริ่มระบบ API
        self.init_ui()
        self.apply_styles()
        self.check_model_file()
        self.load_region_config() # Load default configuration on startup
        
    def init_flask_api(self):
        """ตั้งค่าและเริ่ม Flask Server"""
        self.flask_app = Flask(__name__)
        # Pass self (the MonitorApp instance) to Flask's configuration if needed
        self.flask_app.config['MONITOR_APP'] = self 
        
        # สร้าง Route API
        @self.flask_app.route('/status', methods=['GET'])
        def get_status():
            # ใช้ Lock อ่านข้อมูลล่าสุด
            app_instance = self.flask_app.config['MONITOR_APP']
            with app_instance.data_lock:
                return jsonify(app_instance.api_data)
        
        # เริ่ม Thread API
        self.api_thread = ApiThread(self.flask_app)
        self.api_thread.start()

    def check_model_file(self):
        if not os.path.exists(MODEL_PATH):
            self.alert_label.setText(f"FATAL ERROR: Model file '{MODEL_PATH}' not found.")
            self.alert_label.setStyleSheet("background-color: #A30000; color: white; padding: 10px; border-radius: 5px;")
            self.start_webcam_btn.setEnabled(False)
            self.open_file_btn.setEnabled(False)

    def apply_styles(self):
        """Applies dark theme styles to the entire application, including the new QSlider."""
        # --- Global Dark Theme ---
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2e3440; /* Dark background */
                color: #eceff4;
            }
            QLabel {
                color: #eceff4;
            }
            QPushButton {
                background-color: #5e81ac; /* Light blue accent */
                color: white;
                border: 2px solid #5e81ac;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: bold;
                transition: background-color 0.3s;
            }
            QPushButton:hover {
                background-color: #88c0d0; /* Lighter hover */
                border: 2px solid #88c0d0;
            }
            QPushButton:disabled {
                background-color: #4c566a; /* Darker disabled */
                border: 2px solid #4c566a;
                color: #d8dee9;
            }
            /* Styling for the new confidence slider */
            QSlider::groove:horizontal {
                border: 1px solid #4c566a;
                height: 8px;
                background: #4c566a;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #88c0d0;
                border: 1px solid #88c0d0;
                width: 18px;
                margin: -5px 0; 
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #5e81ac;
                border-radius: 4px;
            }
            /* Styling for QRadioButton */
            QRadioButton {
                spacing: 5px;
                color: #eceff4;
                font-weight: bold;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
                border-radius: 7px;
            }
            QRadioButton::indicator:checked {
                background-color: #5e81ac;
                border: 2px solid #88c0d0;
            }
            QRadioButton::indicator:unchecked {
                background-color: #4c566a;
                border: 1px solid #778899;
            }
        """)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- Title ---
        title_label = QLabel("AI Animal Monitoring System")
        title_label.setFont(QFont("Inter", 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #88c0d0; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        # API Info Label
        api_label = QLabel(f"API Running at: http://localhost:{API_PORT}/status")
        api_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        api_label.setStyleSheet("color: #ebcb8b; font-size: 14px;")
        main_layout.addWidget(api_label)

        # --- Top Control Panel (Buttons) ---
        control_panel = QHBoxLayout()
        control_panel.setSpacing(10)
        
        self.start_webcam_btn = QPushButton("Start Webcam")
        self.start_webcam_btn.clicked.connect(lambda: self.start_monitoring(0))
        control_panel.addWidget(self.start_webcam_btn)

        self.open_file_btn = QPushButton("Open Video File")
        self.open_file_btn.clicked.connect(self.open_file)
        control_panel.addWidget(self.open_file_btn)
        
        # New buttons for Save/Load config
        self.save_config_btn = QPushButton("Save Region Config")
        self.save_config_btn.clicked.connect(self.save_region_config)
        control_panel.addWidget(self.save_config_btn)

        self.load_config_btn = QPushButton("Load Default Config")
        self.load_config_btn.clicked.connect(self.load_region_config)
        control_panel.addWidget(self.load_config_btn)
        
        self.clear_line_btn = QPushButton("Clear Region") 
        self.clear_line_btn.clicked.connect(self.clear_region)
        self.clear_line_btn.setEnabled(False)
        control_panel.addWidget(self.clear_line_btn)

        main_layout.addLayout(control_panel)
        
        # --- Mode Selection and Polygon Finalization (New GUI Element) ---
        mode_layout = QHBoxLayout()
        mode_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        mode_title = QLabel("Alert Region Mode:")
        mode_title.setFont(QFont("Inter", 12, QFont.Weight.Bold))
        mode_layout.addWidget(mode_title)

        self.line_mode_btn = QRadioButton("Line Crossing (Click & Drag)")
        self.line_mode_btn.setChecked(True)
        self.line_mode_btn.toggled.connect(lambda: self.select_mode('LINE'))
        mode_layout.addWidget(self.line_mode_btn)

        self.polygon_mode_btn = QRadioButton("Polygon Area (Click to Add Point)")
        self.polygon_mode_btn.toggled.connect(lambda: self.select_mode('POLYGON'))
        mode_layout.addWidget(self.polygon_mode_btn)
        
        # This button is now for manual finish/hint
        self.finish_polygon_btn = QPushButton("Finish Polygon (Right-Click)")
        self.finish_polygon_btn.clicked.connect(self.finish_polygon)
        self.finish_polygon_btn.setEnabled(False) 
        self.finish_polygon_btn.setMaximumWidth(250)
        mode_layout.addWidget(self.finish_polygon_btn)
        
        main_layout.addLayout(mode_layout)

        # --- Confidence Control Slider ---
        confidence_layout = QHBoxLayout()
        confidence_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        confidence_title = QLabel("Confidence Threshold:")
        confidence_title.setFont(QFont("Inter", 12))
        confidence_layout.addWidget(confidence_title)
        
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100) # 0% to 100%
        # Set initial value from current self.confidence
        self.confidence_slider.setValue(int(self.confidence * 100)) 
        self.confidence_slider.setSingleStep(1)
        self.confidence_slider.setToolTip("Set the minimum confidence level for object detection.")
        self.confidence_slider.valueChanged.connect(self.update_confidence_threshold)
        confidence_layout.addWidget(self.confidence_slider)
        
        self.confidence_value_label = QLabel(f"{int(self.confidence * 100)}%")
        self.confidence_value_label.setFont(QFont("Inter", 12, QFont.Weight.Bold))
        self.confidence_value_label.setFixedWidth(50) # Fix width to prevent layout jump
        confidence_layout.addWidget(self.confidence_value_label)
        
        main_layout.addLayout(confidence_layout)

        # --- Video Display and Alert ---
        self.video_label = QLabel("Select an input source to begin monitoring.")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFont(QFont("Inter", 16))
        
        # Visual style for the video box
        self.video_label.setStyleSheet("background-color: #434c5e; border: 2px solid #5e81ac; border-radius: 10px;")
        self.video_label.setMinimumSize(800, 500)
        
        # Enable mouse tracking and assign custom event handlers
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self.mouse_press
        self.video_label.mouseMoveEvent = self.mouse_move
        self.video_label.mouseReleaseEvent = self.mouse_release
        
        main_layout.addWidget(self.video_label)

        # --- Status/Alert Bar ---
        self.alert_label = QLabel("Status: Ready to monitor.")
        self.alert_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_label.setFont(QFont("Inter", 18, QFont.Weight.Bold))
        self.alert_label.setStyleSheet("background-color: #50fa7b; color: #2e3440; padding: 12px; border-radius: 6px;")
        main_layout.addWidget(self.alert_label)

    # --- New Methods for Mode/Region Control ---
    
    def save_region_config(self):
        """Saves the current region configuration to a JSON file."""
        config = {
            'region_mode': self.region_mode,
            'confidence': self.confidence, # <--- SAVING CONFIDENCE
            # Convert tuples to lists for JSON serialization
            'P1': list(self.P1) if self.P1 else None,
            'P2': list(self.P2) if self.P2 else None,
            'polygon_points': [list(p) for p in self.polygon_points]
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            self.alert_label.setText(f"Config saved to {CONFIG_FILE}.")
            self.alert_label.setStyleSheet("background-color: #8fbcbb; color: #2e3440; padding: 12px; border-radius: 6px;")
        except Exception as e:
            print(f"Error saving config: {e}")
            self.alert_label.setText("Error saving config.")
        
    def load_region_config(self):
        """Loads the region configuration and confidence threshold from a JSON file."""
        if not os.path.exists(CONFIG_FILE):
            print(f"Config file {CONFIG_FILE} not found. Using current settings.")
            return
            
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                
            # --- Load Confidence Threshold ---
            new_confidence = config.get('confidence', DEFAULT_CONFIDENCE_THRESHOLD)
            self.confidence = new_confidence
            # Update the slider and label to reflect the loaded confidence
            self.confidence_slider.setValue(int(self.confidence * 100))
            self.confidence_value_label.setText(f"{int(self.confidence * 100)}%")
            # -----------------------------------
                
            self.region_mode = config.get('region_mode', 'LINE')
            
            # Convert lists back to tuples if they exist
            p1_list = config.get('P1')
            p2_list = config.get('P2')
            self.P1 = tuple(p1_list) if p1_list else None
            self.P2 = tuple(p2_list) if p2_list else None
            
            poly_list = config.get('polygon_points', [])
            self.polygon_points = [tuple(p) for p in poly_list]
            
            # Update UI elements to reflect loaded state
            if self.region_mode == 'LINE':
                self.line_mode_btn.setChecked(True)
            else:
                self.polygon_mode_btn.setChecked(True)
            
            # Update polygon button status
            if self.region_mode == 'POLYGON' and len(self.polygon_points) >= 3:
                self.finish_polygon_btn.setText(f"Polygon Locked ({len(self.polygon_points)} pts)")
                self.finish_polygon_btn.setEnabled(False)
            else:
                self.finish_polygon_btn.setText("Finish Polygon (Right-Click)")
                self.finish_polygon_btn.setEnabled(False)
                
            # Clear any temporary drawing state
            self.temp_P = None
            self.is_drawing = False

            # Update API mode immediately
            with self.data_lock:
                 self.api_data['mode'] = self.region_mode

            self.alert_label.setText(f"Config loaded from {CONFIG_FILE}.")
            self.alert_label.setStyleSheet("background-color: #a3be8c; color: #2e3440; padding: 12px; border-radius: 6px;")
            
            print(f"Config loaded: Mode={self.region_mode}, Conf={self.confidence:.2f}")
            
        except Exception as e:
            print(f"Error loading config: {e}")
            self.alert_label.setText("Error loading config.")


    def select_mode(self, mode):
        """Switches the active region drawing mode."""
        if self.region_mode != mode:
            self.region_mode = mode
            self.clear_region() # Clear any drawn region when switching modes

        if mode == 'POLYGON':
            self.finish_polygon_btn.setText("Finish Polygon (Right-Click)")
            self.finish_polygon_btn.setEnabled(False)
        elif mode == 'LINE':
            self.finish_polygon_btn.setText("Finish Polygon (Right-Click)")
            self.finish_polygon_btn.setEnabled(False)

    def finish_polygon(self):
        """Finalizes the polygon drawing, if enough points exist."""
        if self.region_mode == 'POLYGON' and len(self.polygon_points) >= 3:
            self.temp_P = None # Clear temporary point
            self.is_drawing = False
            self.finish_polygon_btn.setText(f"Polygon Locked ({len(self.polygon_points)} pts)")
            self.finish_polygon_btn.setEnabled(False)
            print(f"Polygon finalized with {len(self.polygon_points)} points.")
        else:
            print("Cannot finalize polygon: need at least 3 points.")

    def clear_region(self):
        """Clears all region data (line points or polygon points)."""
        self.P1 = None
        self.P2 = None
        self.polygon_points = []
        self.temp_P = None
        self.is_drawing = False
        self.handle_breach(False) # Clear alert
        self.finish_polygon_btn.setText("Finish Polygon (Right-Click)")
        self.finish_polygon_btn.setEnabled(False)
        print("Control region cleared.")

    # --- New Method for Slider ---
    def update_confidence_threshold(self, value):
        """Updates the confidence threshold based on the slider value (0-100)."""
        new_confidence = value / 100.0
        self.confidence = new_confidence
        self.confidence_value_label.setText(f"{value}%")

    # --- Video Thread Management ---
    
    def start_monitoring(self, source):
        if self.video_running:
            self.stop_monitoring()
        
        self.video_running = True
        self.video_thread = VideoThread(source)
        # Pass current state to the thread when starting
        self.video_thread.confidence = self.confidence
        self.video_thread.region_mode = self.region_mode
        self.video_thread.polygon_points = self.polygon_points
        self.video_thread.P1 = self.P1
        self.video_thread.P2 = self.P2
        
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.breach_signal.connect(self.handle_breach)
        # เชื่อมต่อ Signal ข้อมูล API
        self.video_thread.detection_data_signal.connect(self.update_api_data)
        
        self.video_thread.start()
        
        self.clear_line_btn.setEnabled(True)
        self.alert_label.setText("Status: Monitoring started.")
        self.alert_label.setStyleSheet("background-color: #5e81ac; color: white; padding: 12px; border-radius: 6px;")
        
        # Universal Stop Button Logic
        self.start_webcam_btn.setText("Stop Monitoring")
        self.start_webcam_btn.clicked.disconnect()
        self.start_webcam_btn.clicked.connect(self.stop_monitoring)
        self.open_file_btn.setEnabled(False) 

    def stop_monitoring(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        
        self.clear_region() 
        self.video_running = False
        
        # Reset API Data
        with self.data_lock:
            self.api_data["status"] = "stopped"
            self.api_data["is_breach"] = False
            self.api_data["detections"] = []
        
        # Reset webcam button to original function
        self.start_webcam_btn.setText("Start Webcam")
        self.start_webcam_btn.clicked.disconnect()
        self.start_webcam_btn.clicked.connect(lambda: self.start_monitoring(0))
        
        self.open_file_btn.setEnabled(True)
        self.clear_line_btn.setEnabled(False)
        
        # Clear the video label to remove the last drawn frame/line
        self.video_label.setPixmap(QPixmap()) 
        self.video_label.setText("Select an input source to begin monitoring.")
        
        self.alert_label.setText("Status: Monitoring stopped.")
        self.alert_label.setStyleSheet("background-color: #4c566a; color: white; padding: 12px; border-radius: 6px;")

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", 
                                                   "Video Files (*.mp4 *.avi *.mov);;All Files (*)")
        if file_name:
            self.start_monitoring(file_name)

    # --- UI Update and Alert Handling ---

    def update_image(self, cv_img):
        """Converts an OpenCV image to QPixmap and displays it."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(convert_to_Qt_format)
        
        # Display the image, scaled to fit the label while keeping aspect ratio
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), 
                                                 Qt.AspectRatioMode.KeepAspectRatio, 
                                                 Qt.TransformationMode.SmoothTransformation))
        
        # Pass the line coordinates, polygon points, and mode to the worker thread for the next frame
        if self.video_thread:
            self.video_thread.P1 = self.P1
            self.video_thread.P2 = self.P2
            self.video_thread.polygon_points = self.polygon_points
            self.video_thread.temp_P = self.temp_P # For drawing preview
            self.video_thread.region_mode = self.region_mode
            self.video_thread.confidence = self.confidence 
            
    def handle_breach(self, is_breach):
        """Updates the alert label based on the breach signal."""
        if is_breach:
            self.alert_label.setText(ALERT_TEXT)
            self.alert_label.setStyleSheet("background-color: #bf616a; color: white; padding: 12px; border-radius: 6px;")
        elif self.video_running:
            self.alert_label.setText("Status: Monitoring...")
            self.alert_label.setStyleSheet("background-color: #a3be8c; color: #2e3440; padding: 12px; border-radius: 6px;")

    # --- เพิ่ม: ฟังก์ชันอัปเดตข้อมูลสำหรับ API ---
    def update_api_data(self, detection_list):
        is_breach = any(d['is_breaching'] for d in detection_list)
        with self.data_lock:
            self.api_data["status"] = "monitoring"
            self.api_data["is_breach"] = is_breach
            self.api_data["detections"] = detection_list
            self.api_data["timestamp"] = time.time()

    # --- Line/Polygon Drawing Logic (Improved Robustness) ---

    def map_mouse_to_frame(self, pos):
        """
        Maps mouse position on the scaled QLabel to the actual frame resolution.
        This handles the scaling and letterboxing (KeepAspectRatio).
        """
        pixmap = self.video_label.pixmap()
        if not pixmap:
            return None
            
        original_w = pixmap.width()
        original_h = pixmap.height()
        
        if original_w == 0 or original_h == 0:
            return None
            
        label_w = self.video_label.width()
        label_h = self.video_label.height()

        # Calculate scaling factor
        scale_w = label_w / original_w
        scale_h = label_h / original_h
        scale = min(scale_w, scale_h)
        
        scaled_w = int(original_w * scale)
        scaled_h = int(original_h * scale)
        
        # Calculate offsets (pillar-box or letter-box)
        offset_x = (label_w - scaled_w) // 2
        offset_y = (label_h - scaled_h) // 2
        
        # Check if click is outside the MAPPED IMAGE area (in the letterbox/pillarbox)
        if pos.x() < offset_x or pos.x() >= offset_x + scaled_w or \
           pos.y() < offset_y or pos.y() >= offset_y + scaled_h:
            return None
            
        # Map mouse coordinates back to original frame coordinates
        frame_x = int((pos.x() - offset_x) / scale)
        frame_y = int((pos.y() - offset_y) / scale)
        
        # CLAMPING: Ensure coordinates are within the original frame dimensions [0, W-1] and [0, H-1]
        frame_x = max(0, min(frame_x, original_w - 1))
        frame_y = max(0, min(frame_y, original_h - 1))
        
        return (frame_x, frame_y)

    def mouse_press(self, event):
        if not self.video_running: return
        mapped_point = self.map_mouse_to_frame(event.pos())
        if not mapped_point: return
        
        if self.region_mode == 'LINE':
            if event.button() == Qt.MouseButton.LeftButton:
                self.is_drawing = True
                self.P1 = mapped_point
                self.P2 = None 
                # print(f"Line drawing started at: {self.P1}")

        elif self.region_mode == 'POLYGON':
            if event.button() == Qt.MouseButton.LeftButton:
                # Add a point to the polygon
                self.polygon_points.append(mapped_point)
                
                # Update button and print status
                if len(self.polygon_points) >= 1:
                    self.finish_polygon_btn.setEnabled(True)
                    self.finish_polygon_btn.setText(f"Add Points ({len(self.polygon_points)} pts)")
                print(f"Polygon point added: {mapped_point}")
                
    def mouse_move(self, event):
        if not self.video_running: return
        mapped_point = self.map_mouse_to_frame(event.pos())
        if not mapped_point: return
        
        if self.region_mode == 'LINE' and self.is_drawing:
            self.P2 = mapped_point
        
        elif self.region_mode == 'POLYGON':
            # Use temp_P to draw a line from the last point to the cursor for preview
            if self.polygon_points:
                self.temp_P = mapped_point

    def mouse_release(self, event):
        if not self.video_running: return
        mapped_point = self.map_mouse_to_frame(event.pos())
        if not mapped_point: return
        
        if self.region_mode == 'LINE' and self.is_drawing:
            self.is_drawing = False
            self.P2 = mapped_point
            if self.P1 and self.P2:
                 print(f"Line locked between {self.P1} and {self.P2}")
                 
        elif self.region_mode == 'POLYGON':
            if event.button() == Qt.MouseButton.RightButton and len(self.polygon_points) >= 3:
                # Finalize the polygon on Right-Click
                self.finish_polygon()
                # Clear temp point immediately to stop the preview line drawing
                self.temp_P = None


    def closeEvent(self, event):
        """Ensure the video thread is stopped when closing the window."""
        self.stop_monitoring()
        event.accept()

# --- 6. EXECUTION ---

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Check if the mandatory library is installed
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except ImportError:
        print("FATAL ERROR: PyQt6 library not found.")
        print("Please install it using: pip install PyQt6")
        sys.exit(1)

    window = MonitorApp()
    window.show()

    sys.exit(app.exec())
