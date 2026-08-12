# ============================================================================
# PROFESSIONAL HAND GESTURE CLASSIFICATION SYSTEM
# Real-time detection with every hand movement
# ============================================================================

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
from threading import Thread
from collections import deque

# ============================================================================
# STREAMLIT CONFIGURATION (Must be first)
# ============================================================================
st.set_page_config(
    page_title="Professional Hand Gesture Classifier",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR PROFESSIONAL LOOK
# ============================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .predicted-letter {
        font-size: 5rem;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .confidence-text {
        font-size: 1.5rem;
        color: #E0E0E0;
        margin-top: 1rem;
    }
    .status-success {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-warning {
        color: #FF9800;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING WITH ERROR HANDLING
# ============================================================================
@st.cache_resource
def load_model_inception():
    """Load InceptionV3 model with comprehensive error handling"""
    try:
        with st.spinner("🔄 Loading AI model..."):
            model = tf.keras.models.load_model("inceptionv3_finetuned.keras")
        st.success("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'inceptionv3_finetuned.keras' not found!")
        st.info("Please ensure the model file is in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model = load_model_inception()

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================
IMG_SIZE = 299
CLASS_MAPPING = {i: chr(65 + i) for i in range(26)}  # 0→A, 1→B, ..., 25→Z
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to display prediction
SMOOTHING_WINDOW = 5  # Number of frames for prediction smoothing

# ============================================================================
# PREDICTION SMOOTHING CLASS
# ============================================================================
class PredictionSmoother:
    """Smooths predictions over multiple frames to reduce jitter"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
    
    def add_prediction(self, letter, confidence):
        """Add new prediction to the smoothing window"""
        self.predictions.append(letter)
        self.confidences.append(confidence)
    
    def get_smoothed_prediction(self):
        """Get the most common prediction in the window"""
        if not self.predictions:
            return None, 0.0
        
        # Count occurrences of each letter
        letter_counts = {}
        for letter, conf in zip(self.predictions, self.confidences):
            if letter not in letter_counts:
                letter_counts[letter] = {'count': 0, 'total_conf': 0.0}
            letter_counts[letter]['count'] += 1
            letter_counts[letter]['total_conf'] += conf
        
        # Get most common letter
        best_letter = max(letter_counts.items(), 
                         key=lambda x: (x[1]['count'], x[1]['total_conf']))
        
        avg_confidence = best_letter[1]['total_conf'] / best_letter[1]['count']
        return best_letter[0], avg_confidence
    
    def reset(self):
        """Clear the smoothing window"""
        self.predictions.clear()
        self.confidences.clear()

# ============================================================================
# IMAGE PREPROCESSING FUNCTIONS
# ============================================================================
def preprocess_frame(frame):
    """
    Professional preprocessing pipeline for InceptionV3
    
    Args:
        frame: Input frame (BGR or RGB)
    
    Returns:
        Preprocessed tensor ready for model prediction
    """
    # Convert BGR to RGB if needed
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # Check if it's BGR (OpenCV format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rgb_frame = frame
    
    # Resize to InceptionV3 input size with high-quality interpolation
    resized = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE), 
                        interpolation=cv2.INTER_CUBIC)
    
    # Normalize to [0, 1] range
    normalized = resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    batched = np.expand_dims(normalized, axis=0)
    
    return batched

def predict_image(image):
    """
    Predict hand gesture from image with enhanced preprocessing
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        class_idx: Predicted class index
        confidence: Prediction confidence
        all_predictions: Full prediction array
    """
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
    
    # Preprocess
    processed = preprocess_frame(img)
    
    # Predict with model
    predictions = model.predict(processed, verbose=0)
    
    # Extract results
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    
    return class_idx, confidence, predictions[0]

def draw_professional_overlay(frame, letter, confidence, fps, is_detecting):
    """
    Draw professional overlay with prediction on frame
    
    Args:
        frame: Input video frame
        letter: Predicted letter
        confidence: Confidence score
        fps: Current frames per second
        is_detecting: Whether hand is being detected
    
    Returns:
        Frame with overlay
    """
    display_frame = frame.copy()
    h, w = display_frame.shape[:2]
    
    # Create semi-transparent overlay
    overlay = display_frame.copy()
    
    # Top bar with dark background
    cv2.rectangle(overlay, (0, 0), (w, 120), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)
    
    # Status indicator
    status_color = (0, 255, 0) if is_detecting else (0, 165, 255)
    status_text = "DETECTING" if is_detecting else "READY"
    cv2.circle(display_frame, (30, 40), 12, status_color, -1)
    cv2.putText(display_frame, status_text, (55, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # FPS Counter
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (w - 150, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Main prediction display
    if confidence >= CONFIDENCE_THRESHOLD:
        # Large prediction box
        box_height = 200
        box_y = h - box_height - 20
        
        # Semi-transparent background
        overlay2 = display_frame.copy()
        cv2.rectangle(overlay2, (20, box_y), (w - 20, h - 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Border
        cv2.rectangle(display_frame, (20, box_y), (w - 20, h - 20), (0, 255, 0), 3)
        
        # Predicted letter (very large)
        text_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_BOLD, 5, 8)[0]
        text_x = (w - text_size[0]) // 2
        text_y = box_y + 100
        cv2.putText(display_frame, letter, (text_x, text_y), 
                    cv2.FONT_HERSHEY_BOLD, 5, (0, 255, 0), 8)
        
        # Confidence bar
        bar_width = w - 80
        bar_x = 40
        bar_y = h - 60
        
        # Background bar
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 30), 
                     (60, 60, 60), -1)
        
        # Confidence fill
        fill_width = int(bar_width * confidence)
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + 30), 
                     color, -1)
        
        # Confidence text
        conf_text = f"{confidence:.1%}"
        cv2.putText(display_frame, conf_text, (bar_x + bar_width + 15, bar_y + 23), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        # "Make a gesture" prompt
        prompt = "Position your hand in view"
        text_size = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(display_frame, prompt, (text_x, h - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    return display_frame

# ============================================================================
# STREAMLIT UI LAYOUT
# ============================================================================
st.markdown('<div class="main-header">🖐️ Professional Hand Gesture Classifier</div>', 
            unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")

# Input method selection
input_method = st.sidebar.radio(
    "📥 Input Method",
    ["Real-Time Webcam", "Upload Image"],
    help="Choose how to provide hand gesture images"
)

if input_method == "Real-Time Webcam":
    st.sidebar.markdown("### 🎥 Webcam Settings")
    
    # Start/Stop button
    start_webcam = st.sidebar.button("▶️ Start Detection", type="primary", use_container_width=True)
    stop_webcam = st.sidebar.button("⏹️ Stop Detection", use_container_width=True)
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        mirror_video = st.checkbox("Mirror Video", value=True)
        show_fps = st.checkbox("Show FPS", value=True)
        enable_smoothing = st.checkbox("Enable Smoothing", value=True, 
                                       help="Reduces jitter in predictions")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    
    # Update global threshold
    CONFIDENCE_THRESHOLD = confidence_threshold
    
    # Performance metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Performance")
    fps_metric = st.sidebar.empty()
    frames_metric = st.sidebar.empty()

# Info section
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ System Info")
st.sidebar.info(f"""
**Model:** InceptionV3 Fine-tuned  
**Input Size:** {IMG_SIZE}×{IMG_SIZE}  
**Classes:** 26 (A-Z)  
**Mode:** Real-time Detection
""")

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Instructions")
st.sidebar.markdown("""
1. Click **Start Detection**
2. Allow camera access
3. Position hand clearly in view
4. Make hand gesture (A-Z)
5. System detects automatically
6. Click **Stop** when done
""")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

if input_method == "Upload Image":
    # ========================================================================
    # IMAGE UPLOAD MODE
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of your hand gesture"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("🎯 Prediction Result")
        
        if uploaded_file and st.button("🔍 Analyze Gesture", type="primary"):
            with st.spinner("Analyzing hand gesture..."):
                class_idx, confidence, all_preds = predict_image(image)
                predicted_letter = CLASS_MAPPING[class_idx]
                
                # Display result
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="predicted-letter">{predicted_letter}</div>
                    <div class="confidence-text">Confidence: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Top 5 predictions
                st.markdown("### 📊 Top 5 Predictions")
                top_5_idx = np.argsort(all_preds)[-5:][::-1]
                
                for idx in top_5_idx:
                    letter = CLASS_MAPPING[idx]
                    conf = all_preds[idx]
                    st.progress(float(conf))
                    st.text(f"{letter}: {conf:.1%}")

else:
    # ========================================================================
    # REAL-TIME WEBCAM MODE
    # ========================================================================
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("🎯 Current Detection")
        prediction_placeholder = st.empty()
        confidence_display = st.empty()
        
        st.markdown("### 📈 Detection History")
        history_placeholder = st.empty()
    
    # Initialize session state for webcam control
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    
    if start_webcam:
        st.session_state.webcam_active = True
    
    if stop_webcam:
        st.session_state.webcam_active = False
    
    # ========================================================================
    # WEBCAM PROCESSING LOOP
    # ========================================================================
    if st.session_state.webcam_active:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access webcam!")
            st.error("""
            **Troubleshooting:**
            - Check camera connection
            - Close other apps using camera
            - Grant browser camera permissions
            - Try refreshing the page
            """)
            st.session_state.webcam_active = False
        else:
            # Set camera properties for optimal performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            # Initialize prediction smoother
            smoother = PredictionSmoother(window_size=SMOOTHING_WINDOW)
            
            # Performance tracking
            prev_time = time.time()
            frame_count = 0
            detection_history = deque(maxlen=10)
            
            st.success("✅ Camera connected! Detection active...")
            
            # Main processing loop
            while st.session_state.webcam_active:
                ret, frame = cap.read()
                
                if not ret:
                    st.warning("⚠️ Failed to read frame")
                    break
                
                # Mirror video if enabled
                if mirror_video:
                    frame = cv2.flip(frame, 1)
                
                # Predict gesture from current frame
                class_idx, confidence, all_preds = predict_image(frame)
                predicted_letter = CLASS_MAPPING[class_idx]
                
                # Apply smoothing if enabled
                if enable_smoothing:
                    smoother.add_prediction(predicted_letter, confidence)
                    display_letter, display_conf = smoother.get_smoothed_prediction()
                else:
                    display_letter, display_conf = predicted_letter, confidence
                
                # Track detection history
                if display_conf >= CONFIDENCE_THRESHOLD:
                    detection_history.append(f"{display_letter} ({display_conf:.0%})")
                
                # Calculate FPS
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                # Draw overlay on frame
                is_detecting = display_conf >= CONFIDENCE_THRESHOLD
                annotated_frame = draw_professional_overlay(
                    frame, display_letter, display_conf, fps, is_detecting
                )
                
                # Convert to RGB for Streamlit
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display video
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                # Update prediction display
                if is_detecting:
                    prediction_placeholder.markdown(f"""
                    <div class="prediction-box">
                        <div class="predicted-letter">{display_letter}</div>
                        <div class="confidence-text">{display_conf:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    confidence_display.progress(float(display_conf))
                else:
                    prediction_placeholder.info("👋 Make a hand gesture")
                    confidence_display.empty()
                
                # Update detection history
                if detection_history:
                    history_text = "\n".join([f"• {h}" for h in list(detection_history)[-5:]])
                    history_placeholder.text(history_text)
                
                # Update metrics
                frame_count += 1
                fps_metric.metric("FPS", f"{fps:.1f}")
                frames_metric.metric("Frames", frame_count)
                
                # Small delay
                time.sleep(0.01)
            
            # Cleanup
            cap.release()
            st.info("📷 Detection stopped")
            smoother.reset()
    else:
        video_placeholder.info("👆 Click 'Start Detection' in the sidebar to begin")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <strong>Professional Hand Gesture Recognition System</strong><br>
    Powered by TensorFlow InceptionV3 | Real-time AI Detection<br>
    💡 <em>Best results with good lighting and clear hand visibility</em>
</div>
""", unsafe_allow_html=True)