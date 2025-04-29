import streamlit as st
import cv2
import base64
import json
import os
import tempfile
import time
from dotenv import load_dotenv
import threading
from queue import Queue
from openai import OpenAI
from ultralytics import YOLO

st.title("Real-Time Action Recognition")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# With these:
# For local development with .env file
if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
# For Streamlit Cloud deployment
else:
    api_key = st.secrets["api_keys"]["openai"]

client = OpenAI(api_key=api_key)

# Configuration - adjust these values to balance speed and accuracy
YOLO_INTERVAL = 5 # Process every 5th frame with YOLO
DISPLAY_INTERVAL = 5  # Update display every 3rd frame
API_INTERVAL = 5  # Send to API every 45th frame
MAX_WORKERS = 3 # Maximum number of concurrent API calls

# Load YOLOv8 model with caching
@st.cache_resource
def load_model():
    # Use a smaller model for better speed
    return YOLO("yolov8n.pt")  # 'n' for nano is much faster than 's' for small

# Global variables
model = load_model()
current_detections = []
current_actions = []
api_queue = Queue(maxsize=MAX_WORKERS)
stop_event = threading.Event()

def extract_frame_base64(frame):
    """Convert frame to base64 for OpenAI API."""
    # Resize frame to reduce API costs and improve speed
    resized_frame = cv2.resize(frame, (640, 360))
    _, buffer = cv2.imencode(".jpg", resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode("utf-8")

def api_worker():
    """Background thread to process API calls."""
    global current_actions
    while not stop_event.is_set():
        try:
            if not api_queue.empty():
                frame_base64 = api_queue.get()
                
                # Send to OpenAI API
                prompt_messages = [
                    {
                        "role": "user",
                        "content": [
                            "Detect people actions only washing hand , wearing head mask, wearing face mask , walking  . Return JSON where each item has (action). Be very brief.",
                            {"image": frame_base64, "resize": 640},
                        ],
                    }
                ]
                
                try:
                    result = client.chat.completions.create(
                        model="gpt-4.5-preview",
                        messages=prompt_messages,
                        max_tokens=50,  # Reduced for faster response
                        temperature=0.7,  # Lower temperature for more concise responses
                    )
                    
                    response_text = result.choices[0].message.content
                    predictions = json.loads(response_text)
                    if isinstance(predictions, list):
                        current_actions = predictions
                except Exception as e:
                    print(f"API Error: {str(e)}")
            
            # Small delay to prevent CPU hogging
            time.sleep(0.1)
        except Exception as e:
            print(f"Worker error: {str(e)}")
            time.sleep(0.5)

def draw_predictions(frame, detections, actions):
    """Draw bounding boxes and action labels."""
    # Match detections with available actions
    for i, det in enumerate(detections):
        try:
            x1, y1, x2, y2 = map(int, det[:4])
            
            # Get action if available
            action_text = "Unknown"
            if i < len(actions):
                action_text = actions[i].get("action", "Unknown")
                
            # Simplified drawing for better performance
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, action_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            print(f"Drawing error: {str(e)}")
            
    return frame

def process_video(video_path):
    """Process video with optimized pipeline."""
    global current_detections, current_actions
    
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    # Start API worker thread
    api_thread = threading.Thread(target=api_worker, daemon=True)
    api_thread.start()
    
    frame_count = 0
    last_time = time.time()
    fps_display = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - last_time
        if elapsed > 0:
            fps_display = 0.9 * fps_display + 0.1 * (1 / elapsed)  # Smoothed FPS
        last_time = current_time
        
        # Update progress
        if frame_count % DISPLAY_INTERVAL == 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
        
        # Run YOLO detection at specified intervals
        if frame_count % YOLO_INTERVAL == 0:
            # Process with YOLO using a smaller input size
            results = model(frame, conf=0.5, imgsz=640)
            
            # Update detections
            current_detections = []
            for result in results:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = box.tolist()
                    if int(cls) == 0:  # Class 0 is 'person' in COCO dataset
                        current_detections.append([x1, y1, x2, y2])
        
        # Submit frames to API queue at specified intervals
        if frame_count % API_INTERVAL == 0 and current_detections and api_queue.qsize() < MAX_WORKERS:
            frame_base64 = extract_frame_base64(frame)
            api_queue.put(frame_base64)
        
        # Update display at specified intervals
        if frame_count % DISPLAY_INTERVAL == 0:
            display_frame = frame.copy()
            
            # Draw predictions
            display_frame = draw_predictions(display_frame, current_detections, current_actions)
            
            # Add FPS counter
            cv2.putText(display_frame, f"FPS: {fps_display:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            stframe.image(display_frame, channels="BGR", use_container_width=True)
        
        frame_count += 1
    
    # Clean up
    cap.release()
    stop_event.set()
    api_thread.join(timeout=2)
    
    st.success("Video processing complete!")

# Main app
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name
    
    # Process video
    st.info("Processing video... This may take a moment.")
    process_video(video_path)
    
    # Clean up temp file
    try:
        os.unlink(video_path)
    except:
        pass