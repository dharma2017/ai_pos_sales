import streamlit as st
import cv2
import pandas as pd
from datetime import datetime
import numpy as np
from pathlib import Path
import time
import socket
from ultralytics import YOLO
import torch
from PIL import Image

# Configuration
EXCEL_FILE = "sales_data.xlsx"
DETECTION_THRESHOLD = 0.5

# Initialize YOLO model (download happens automatically on first run)
@st.cache_resource
def load_detection_model():
    """Load YOLOv8 model for cup detection"""
    try:
        # YOLOv8n (nano) is fast and works on CPU
        model = YOLO('yolov8n.pt')  # Will auto-download on first run
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Unable to detect"

def test_camera(index):
    """Test if a camera index is available"""
    try:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            return ret
        return False
    except:
        return False

def find_available_cameras():
    """Find all available camera indices"""
    available = []
    for i in range(10):
        if test_camera(i):
            available.append(i)
    return available

def load_or_create_excel():
    """Load existing Excel file or create new one"""
    if Path(EXCEL_FILE).exists():
        return pd.read_excel(EXCEL_FILE)
    else:
        return pd.DataFrame(columns=['Date', 'Time', 'Item', 'Quantity', 'Total_Cups'])

def save_to_excel(df):
    """Save dataframe to Excel"""
    df.to_excel(EXCEL_FILE, index=False)

def detect_cups_ml(frame, model, confidence_threshold=0.5):
    """
    Detect cups using YOLOv8 model
    COCO dataset classes include 'cup' (class 41)
    """
    if model is None:
        return 0, frame
    
    try:
        # Run inference
        results = model(frame, conf=confidence_threshold, verbose=False)
        
        # Extract detections
        cups_detected = 0
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class ID and name
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                confidence = float(box.conf[0])
                
                # COCO classes: cup=41, bottle=39, wine glass=40, bowl=45
                # You can adjust these based on what you want to detect
                if class_name in ['cup', 'bottle', 'bowl', 'wine glass']:
                    cups_detected += 1
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Color based on item type
                    color = (0, 255, 0)  # Green for cups
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return cups_detected, annotated_frame
    
    except Exception as e:
        st.error(f"Detection error: {e}")
        return 0, frame

# Streamlit UI
st.set_page_config(page_title="AI Cup Counter", layout="wide")
st.title("🤖 AI-Powered Tea/Coffee Shop Sales Tracker")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model status
    st.subheader("🧠 AI Model")
    with st.spinner("Loading detection model..."):
        model = load_detection_model()
    
    if model is not None:
        st.success("✅ YOLOv8 Model Loaded")
        st.info("🎯 Detects: Cups, Bottles, Bowls, Wine Glasses")
    else:
        st.error("❌ Model loading failed")
        st.warning("Install: `pip install ultralytics`")
    
    # Network Info
    st.subheader("📡 Network Information")
    local_ip = get_local_ip()
    st.info(f"Your IP: {local_ip}")
    
    # Camera Selection
    st.subheader("📷 Camera Setup")
    
    if st.button("🔍 Find Available Cameras"):
        with st.spinner("Testing camera connections..."):
            available = find_available_cameras()
            if available:
                st.success(f"✅ Found cameras at indices: {', '.join(map(str, available))}")
                st.session_state.available_cameras = available
            else:
                st.error("❌ No local cameras found. Try IP camera option.")
                st.session_state.available_cameras = []
    
    camera_type = st.radio("Select Camera Type:", 
                          ["Local Webcam", "IP Camera (WiFi)"])
    
    camera_source = None
    
    if camera_type == "Local Webcam":
        if 'available_cameras' in st.session_state and st.session_state.available_cameras:
            st.info(f"✅ Available cameras: {', '.join(map(str, st.session_state.available_cameras))}")
            camera_index = st.selectbox("Select Camera Index:", st.session_state.available_cameras)
        else:
            st.warning("Click 'Find Available Cameras' button above")
            camera_index = st.number_input("Or manually enter Camera Index:", 0, 10, 0)
        camera_source = int(camera_index)
        
    else:  # IP Camera
        st.info("📱 **Setup your phone as camera:**\n"
                "1. Download 'IP Webcam' (Android) or 'EpocCam' (iOS)\n"
                "2. Connect phone to same WiFi\n"
                "3. Start server in app\n"
                "4. Copy URL shown in app\n\n"
                "**Common formats:**\n"
                "- http://192.168.1.100:8080/video\n"
                "- rtsp://192.168.1.100:554/stream")
        camera_url = st.text_input("Enter Camera URL:", "")
        
        if st.button("🧪 Test Connection"):
            if camera_url:
                with st.spinner("Testing connection..."):
                    test_cap = cv2.VideoCapture(camera_url)
                    if test_cap.isOpened():
                        ret, _ = test_cap.read()
                        test_cap.release()
                        if ret:
                            st.success("✅ Connection successful!")
                            camera_source = camera_url
                        else:
                            st.error("❌ Can't read from camera. Check URL format.")
                    else:
                        st.error("❌ Can't connect. Verify URL and WiFi.")
            else:
                st.warning("Please enter a URL first")
        
        if camera_url:
            camera_source = camera_url
    
    # Item selection
    st.subheader("☕ Item Type")
    item_type = st.selectbox("What are you tracking?", ["Tea", "Coffee", "Both"])
    
    # Detection settings
    st.subheader("🎯 AI Detection Settings")
    confidence = st.slider("Detection Confidence:", 0.1, 0.9, 0.5, 0.05)
    st.caption("Higher = fewer false positives, Lower = more detections")
    
    auto_save_threshold = st.number_input("Auto-save when cups detected:", 1, 10, 3)
    st.caption("Automatically save count when this many cups detected")

# Main area - tabs
tab1, tab2, tab3 = st.tabs(["🎥 Live AI Detection", "📊 Sales Data", "ℹ️ Instructions"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live AI Camera Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        detection_info = st.empty()
    
    with col2:
        st.subheader("Controls")
        start_btn = st.button("▶️ Start AI Detection", type="primary")
        stop_btn = st.button("⏹️ Stop Detection")
        
        if 'last_count' in st.session_state:
            st.metric("Last Detection", st.session_state.last_count)
        
        if st.button("💾 Save Last Count"):
            if 'last_count' in st.session_state and st.session_state.last_count > 0:
                df = load_or_create_excel()
                now = datetime.now()
                new_row = pd.DataFrame({
                    'Date': [now.strftime('%Y-%m-%d')],
                    'Time': [now.strftime('%H:%M:%S')],
                    'Item': [item_type],
                    'Quantity': [st.session_state.last_count],
                    'Total_Cups': [len(df) + 1]
                })
                df = pd.concat([df, new_row], ignore_index=True)
                save_to_excel(df)
                st.success(f"✅ Saved {st.session_state.last_count} cups!")
                st.session_state.last_count = 0
        
        st.divider()
        st.subheader("Manual Entry")
        manual_tea = st.number_input("Tea Cups:", 0, 100, 0, key="manual_tea")
        manual_coffee = st.number_input("Coffee Cups:", 0, 100, 0, key="manual_coffee")
        if st.button("➕ Add Manual Entry"):
            df = load_or_create_excel()
            now = datetime.now()
            if manual_tea > 0:
                new_row = pd.DataFrame({
                    'Date': [now.strftime('%Y-%m-%d')],
                    'Time': [now.strftime('%H:%M:%S')],
                    'Item': ['Tea'],
                    'Quantity': [manual_tea],
                    'Total_Cups': [len(df) + 1]
                })
                df = pd.concat([df, new_row], ignore_index=True)
            if manual_coffee > 0:
                new_row = pd.DataFrame({
                    'Date': [now.strftime('%Y-%m-%d')],
                    'Time': [now.strftime('%H:%M:%S')],
                    'Item': ['Coffee'],
                    'Quantity': [manual_coffee],
                    'Total_Cups': [len(df) + 1]
                })
                df = pd.concat([df, new_row], ignore_index=True)
            save_to_excel(df)
            st.success("✅ Entry added!")
        
        st.divider()
        st.subheader("📈 Quick Stats")
        df = load_or_create_excel()
        if not df.empty:
            today = datetime.now().strftime('%Y-%m-%d')
            today_data = df[df['Date'] == today]
            st.metric("Today's Total", len(today_data))
            if 'Tea' in today_data['Item'].values:
                st.metric("Tea", today_data[today_data['Item']=='Tea']['Quantity'].sum())
            if 'Coffee' in today_data['Item'].values:
                st.metric("Coffee", today_data[today_data['Item']=='Coffee']['Quantity'].sum())

    # Camera logic
    if start_btn:
        st.session_state.camera_running = True
    if stop_btn:
        st.session_state.camera_running = False
    
    if st.session_state.get('camera_running', False):
        if camera_source is None:
            status_placeholder.error("❌ Please configure camera source in the sidebar first!")
            st.session_state.camera_running = False
        elif model is None:
            status_placeholder.error("❌ AI model not loaded. Install ultralytics package.")
            st.session_state.camera_running = False
        else:
            try:
                status_placeholder.info("🔄 Connecting to camera...")
                
                cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW if isinstance(camera_source, int) else cv2.CAP_ANY)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(2)
                
                if not cap.isOpened():
                    status_placeholder.error(f"❌ Cannot connect to camera: {camera_source}")
                    st.session_state.camera_running = False
                else:
                    ret, test_frame = cap.read()
                    if not ret:
                        status_placeholder.error("❌ Camera opened but can't read frames")
                        cap.release()
                        st.session_state.camera_running = False
                    else:
                        status_placeholder.success("✅ AI Detection Active!")
                        frame_count = 0
                        last_save_time = time.time()
                        
                        while st.session_state.get('camera_running', False):
                            ret, frame = cap.read()
                            if not ret:
                                status_placeholder.warning("⚠️ Cannot read from camera")
                                break
                            
                            # AI Detection
                            cup_count, annotated_frame = detect_cups_ml(frame, model, confidence)
                            
                            # Store last count
                            st.session_state.last_count = cup_count
                            
                            # Add info overlay
                            cv2.putText(annotated_frame, f'AI Detected: {cup_count} cups', (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f'Confidence: {confidence:.0%}', (10, 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            cv2.putText(annotated_frame, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                       (10, annotated_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.5, (255, 255, 255), 1)
                            
                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                            
                            # Update detection info
                            if cup_count > 0:
                                detection_info.success(f"🎯 Detected {cup_count} cups! (Threshold: {auto_save_threshold})")
                            else:
                                detection_info.info("👀 Scanning for cups...")
                            
                            # Auto-save logic
                            if cup_count >= auto_save_threshold:
                                current_time = time.time()
                                if current_time - last_save_time > 5:  # Save once every 5 seconds
                                    df = load_or_create_excel()
                                    now = datetime.now()
                                    new_row = pd.DataFrame({
                                        'Date': [now.strftime('%Y-%m-%d')],
                                        'Time': [now.strftime('%H:%M:%S')],
                                        'Item': [item_type],
                                        'Quantity': [cup_count],
                                        'Total_Cups': [len(df) + 1]
                                    })
                                    df = pd.concat([df, new_row], ignore_index=True)
                                    save_to_excel(df)
                                    detection_info.success(f"💾 Auto-saved {cup_count} cups!")
                                    last_save_time = current_time
                            
                            frame_count += 1
                            time.sleep(0.03)  # ~30 FPS
                        
                        cap.release()
                        status_placeholder.info("Camera stopped")
            except Exception as e:
                status_placeholder.error(f"❌ Error: {str(e)}")
                st.session_state.camera_running = False

with tab2:
    st.subheader("📊 Sales Records")
    df = load_or_create_excel()
    
    if not df.empty:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            date_filter = st.date_input("Filter by Date", datetime.now())
        with col2:
            item_filter = st.multiselect("Filter by Item", df['Item'].unique(), default=df['Item'].unique())
        
        # Filter data
        filtered_df = df[df['Item'].isin(item_filter)]
        if date_filter:
            filtered_df = filtered_df[filtered_df['Date'] == date_filter.strftime('%Y-%m-%d')]
        
        # Display data
        st.dataframe(filtered_df, use_container_width=True)
        
        # Summary stats
        st.subheader("📈 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sales", filtered_df['Quantity'].sum())
        with col2:
            st.metric("Tea Cups", filtered_df[filtered_df['Item']=='Tea']['Quantity'].sum())
        with col3:
            st.metric("Coffee Cups", filtered_df[filtered_df['Item']=='Coffee']['Quantity'].sum())
        
        # Download button
        if Path(EXCEL_FILE).exists():
            st.download_button(
                "📥 Download Excel",
                data=open(EXCEL_FILE, 'rb').read(),
                file_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("No sales data yet. Start AI detection to see records here.")

with tab3:
    st.markdown("""
    ## 📖 AI-Powered Cup Detection Guide
    
    ### 🤖 AI Model Information
    
    This app uses **YOLOv8** (You Only Look Once) - a state-of-the-art object detection model.
    
    **Detectable Objects:**
    - ☕ Cups (tea cups, coffee cups)
    - 🍶 Bottles
    - 🍷 Wine glasses
    - 🥣 Bowls
    
    The model is trained on the COCO dataset with 80 object classes.
    
    ### 🚀 Setup Instructions
    
    **1. Install Required Packages:**
    ```bash
    pip install streamlit opencv-python pandas openpyxl ultralytics torch pillow
    ```
    
    **2. Camera Setup:**
    
    **Local Webcam:**
    - Click "Find Available Cameras" button
    - Select working camera index (usually 0)
    - Start detection
    
    **WiFi IP Camera (Phone):**
    - Install: IP Webcam (Android) or EpocCam (iOS)
    - Connect phone to same WiFi as computer
    - Start camera server in app
    - Copy URL (e.g., `http://192.168.1.100:8080/video`)
    - Paste in "Enter Camera URL" field
    - Test connection
    
    ### 🎯 Detection Tips
    
    **For Best Results:**
    - 💡 Good lighting is crucial
    - 📐 Position camera at 45° angle above serving area
    - 📏 Keep cups within 1-3 meters of camera
    - 🎨 Avoid cluttered backgrounds
    - ⚖️ Adjust confidence slider based on accuracy
    
    **Confidence Settings:**
    - **High (0.7-0.9)**: Fewer false positives, may miss some cups
    - **Medium (0.4-0.6)**: Balanced detection
    - **Low (0.1-0.3)**: More detections, more false positives
    
    ### 💾 Saving Data
    
    **Automatic:**
    - Set "Auto-save when cups detected" threshold
    - System saves when threshold reached
    - Saves once every 5 seconds to avoid duplicates
    
    **Manual:**
    - Use "Save Last Count" button for detected cups
    - Use "Manual Entry" for custom counts
    
    ### 📊 Data Management
    
    - All data saved to `sales_data.xlsx`
    - View in "Sales Data" tab
    - Filter by date and item type
    - Download Excel file anytime
    
    ### 🔧 Troubleshooting
    
    **Model Not Loading:**
    ```bash
    pip install ultralytics torch --upgrade
    ```
    
    **Poor Detection:**
    - Adjust confidence slider
    - Improve lighting
    - Change camera angle
    - Reduce background clutter
    
    **Camera Issues:**
    - Close other apps using camera
    - Try different camera index
    - Use IP camera option
    - Verify WiFi connection
    
    ### 🎓 How YOLOv8 Works
    
    YOLOv8 processes each frame in real-time:
    1. Analyzes entire image at once
    2. Detects objects and their locations
    3. Assigns confidence scores
    4. Filters by confidence threshold
    5. Draws bounding boxes around detected objects
    
    **Performance:**
    - Speed: ~30 FPS on CPU
    - Accuracy: ~90% on COCO dataset
    - Optimized for real-time detection
    
    ### 📱 Recommended Phone Apps
    
    **Android:**
    - IP Webcam (Free, easy setup)
    - DroidCam (Free/Paid)
    
    **iOS:**
    - EpocCam (Free/Paid)
    - iVCam (Free trial)
    
    All apps display the connection URL clearly!
    """)

st.sidebar.markdown("---")
st.sidebar.info("💡 AI Tip: Position camera for clear view of serving area with good lighting")
st.sidebar.caption("Powered by YOLOv8 🤖")