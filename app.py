"""
Main Flask application for the Golf Camera System
Ties together all modules and provides web interface
"""

import time
import threading
from flask import Flask, Response, render_template, jsonify, request

# Import our modular components
from config.settings import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG, ensure_directories,
    P1_CONFIDENCE_THRESHOLD, P10_CONFIDENCE_THRESHOLD,
    MIN_P1_DURATION, SWING_DETECTION_WINDOW, COOLDOWN_PERIOD,
    DEFAULT_RECORDING_DURATION
)
from utils.logger import setup_logger, log_startup, log_success, log_error
from utils.frame_pool import FramePool
from storage.uploader import BackgroundUploader
from camera.camera_manager import CameraManager
from camera.video_recorder import VideoRecorder
from ai.pose_classifier import GolfPoseClassifier

# Initialize logging
logger = setup_logger(__name__)

# Create Flask app
app = Flask(__name__)

class GolfCameraSystem:
    """Main system orchestrator that coordinates all components"""
    
    def __init__(self):
        """Initialize the complete golf camera system"""
        log_startup(logger, "Golf Camera System initialization")
        
        # Ensure directories exist
        ensure_directories()
        
        # Initialize core components
        self.frame_pool = FramePool()
        self.background_uploader = BackgroundUploader()
        self.camera_manager = CameraManager(frame_pool=self.frame_pool)
        self.video_recorder = VideoRecorder(self.camera_manager, self.background_uploader)
        self.pose_classifier = GolfPoseClassifier()
        
        # System state
        self.auto_recording_enabled = False
        self.pose_detection_enabled = True
        self.current_pose_stage = "waiting"  # waiting, ready, recording, cooldown
        self.pose_start_time = None
        
        # Store current user context for auto recording
        self.current_operator_id = None  
        self.current_assignee_id = None  
        self.current_operator_role = None  
        
        # Setup callbacks
        self.video_recorder.on_recording_started = self._on_recording_started
        self.video_recorder.on_recording_stopped = self._on_recording_stopped
        
        # Initialize AI components
        self._initialize_ai()
        
        log_success(logger, "Golf Camera System initialized")
    
    def _initialize_ai(self) -> None:
        """Initialize AI components"""
        try:
            if self.pose_classifier.initialize():
                # ✅ CRITICAL: Disable transition validation like original
                self.pose_classifier.validate_transitions = False
                print("🔧 RESTORED: Transition validation disabled (like original working code)")
                log_success(logger, "AI components initialized - validation disabled")
            else:
                log_error(logger, "GolfCameraSystem._initialize_ai", 
                        Exception("Failed to initialize AI components"))
        except Exception as e:
            log_error(logger, "GolfCameraSystem._initialize_ai", e)
    
    def _on_recording_started(self) -> None:
        """Callback when recording starts"""
        log_success(logger, "Recording started callback triggered")
    
    def _on_recording_stopped(self, upload_success: bool) -> None:
        """Callback when recording stops"""
        log_success(logger, f"Recording stopped callback triggered - upload: {'success' if upload_success else 'failed'}")
        
        # Reset pose stage after recording
        if self.current_pose_stage == "recording":
            self.current_pose_stage = "cooldown"
            self.pose_start_time = time.time()
    
    def set_user_context(self, user_id: str, role: str = None, assignee_id: str = None) -> bool:
        """Set current user context for auto recording"""
        try:
            self.current_operator_id = user_id
            self.current_operator_role = role
            
            # If assignee_id is provided, use it; otherwise default to operator
            self.current_assignee_id = assignee_id if assignee_id else user_id
            
            log_success(logger, f"User context set - Operator: {user_id}, Assignee: {self.current_assignee_id}")
            print(f"✅ Context set - Operator: {user_id}, Videos will go to: {self.current_assignee_id}")
            return True
        except Exception as e:
            log_error(logger, "GolfCameraSystem.set_user_context", e)
            return False
    
    def process_frame_for_streaming(self, frame):
        """Process frame for streaming with pose detection and overlays"""
        try:
            # Process pose detection if enabled
            if self.pose_detection_enabled and self.pose_classifier.model_manager.is_loaded:
                frame, predicted_class, confidence, p1_confidence, p10_confidence = \
                    self.pose_classifier.classify_pose(frame)
                
                # Auto-recording logic
                self._handle_auto_recording_logic(predicted_class, p1_confidence, p10_confidence)
                
                # Add pose information overlays
                self.camera_manager.add_overlay_text(frame, f"Pose: {predicted_class}", (10, 70))
                self.camera_manager.add_overlay_text(frame, f"P1: {p1_confidence:.2f}", (10, 100))
                self.camera_manager.add_overlay_text(frame, f"P10: {p10_confidence:.2f}", (10, 130))
                self.camera_manager.add_overlay_text(frame, f"Stage: {self.current_pose_stage}", (10, 160))
                
                if self.auto_recording_enabled:
                    self.camera_manager.add_overlay_text(frame, "AUTO REC: ON", (10, 190), 
                                                    color=(255, 0, 0))
                
                # 🔧 FIXED: Show user context status
                if self.current_operator_id:
                    self.camera_manager.add_overlay_text(frame, f"User: {self.current_operator_id[:8]}...", (10, 220), 
                                                    color=(0, 255, 0))
                else:
                    self.camera_manager.add_overlay_text(frame, "No User Context", (10, 220), 
                                                    color=(255, 255, 0))
            
            elif not self.pose_classifier.model_manager.is_loaded:
                self.camera_manager.add_overlay_text(frame, "Model Loading...", (10, 70), 
                                                color=(0, 0, 255))
            
            # Add recording indicator if recording
            if self.video_recorder.is_recording:
                self.camera_manager.add_recording_indicator(frame)
            
            return frame
            
        except Exception as e:
            log_error(logger, "GolfCameraSystem.process_frame_for_streaming", e)
            return frame
    
    def _handle_auto_recording_logic(self, predicted_class: str, 
                           p1_confidence: float, p10_confidence: float) -> None:
        """Handle auto-recording logic"""
        
        if not self.auto_recording_enabled:
            return
        
        # Check if we have user context
        if not self.current_operator_id or not self.current_assignee_id:
            print("Incomplete user context - auto recording disabled")
            return
        
        current_time = time.time()
        
        P1_THRESHOLD = 0.7
        P10_THRESHOLD = 0.7
        MIN_P1_DURATION = 1.0
        SWING_DETECTION_WINDOW = 25.0
        
        try:
            if not self.video_recorder.is_recording:
                if self.current_pose_stage == "waiting":
                    if predicted_class == "P1" and p1_confidence > P1_THRESHOLD:
                        self.current_pose_stage = "ready"
                        self.pose_start_time = current_time
                        print(f"🏌️ P1 detected! Ready to record... (conf: {p1_confidence:.2f})")
                
                elif self.current_pose_stage == "ready":
                    time_in_ready = current_time - (self.pose_start_time or 0)
                    
                    if time_in_ready > MIN_P1_DURATION:
                        if predicted_class in ["P1", "P2", "P3"]:
                            self.current_pose_stage = "recording"
                            self.video_recorder.stop_recording_early = False
                            # 🚀 ENHANCED: Pass both operator and assignee
                            success = self.video_recorder.start_recording(
                                user_id=self.current_operator_id,      # Who is recording
                                assignee_id=self.current_assignee_id,  # Who it's for
                                duration=20, 
                                auto_triggered=True
                            )
                            if success:
                                print(f"🎥 Auto recording started - Operator: {self.current_operator_id}, For: {self.current_assignee_id}")
                            else:
                                print("❌ Failed to start auto recording")
                                self.current_pose_stage = "waiting"
                        else:
                            self.current_pose_stage = "waiting"
                            print("⚠️ Lost P1 detection, resetting...")

                    elif predicted_class in ["P2", "P3"] and time_in_ready > 0.5:
                        self.current_pose_stage = "recording"
                        # ✅ CRITICAL FIX: Use stored user_id for early movement detection
                        self.video_recorder.stop_recording_early = False
                        success = self.video_recorder.start_recording(
                            user_id=self.current_user_id,  # ✅ Use stored user_id instead of hardcoded
                            duration=20, 
                            auto_triggered=True
                        )
                        if success:
                            print(f"🎥 Early swing movement detected - Starting recording for user: {self.current_user_id}")
                        else:
                            print("❌ Failed to start early auto recording")
                            self.current_pose_stage = "waiting"
            
            # ✅ CRITICAL: Fixed P10 stop logic with DIRECT ASSIGNMENT ONLY
            elif self.video_recorder.is_recording and self.current_pose_stage == "recording":
                if predicted_class == "P10" and p10_confidence > P10_THRESHOLD:
                    self.current_pose_stage = "cooldown"
                    # ✅ DIRECT ASSIGNMENT like original (not method call)
                    self.video_recorder.stop_recording_early = True
                    print(f"🛑 P10 detected! Signaling early stop... (conf: {p10_confidence:.2f})")
                
                elif current_time - (self.pose_start_time or 0) > SWING_DETECTION_WINDOW:
                    self.current_pose_stage = "cooldown"
                    # ✅ DIRECT ASSIGNMENT like original (not method call)
                    self.video_recorder.stop_recording_early = True
                    print("⏰ Recording timeout - Signaling stop!")
            
            # Reset to waiting after cooldown
            if self.current_pose_stage == "cooldown" and not self.video_recorder.is_recording:
                if current_time - (self.pose_start_time or 0) > 5:
                    self.current_pose_stage = "waiting"
                    print("⏰ Ready for next swing detection")
                    
        except Exception as e:
            log_error(logger, "GolfCameraSystem._handle_auto_recording_logic", e)
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        try:
            classifier_state = self.pose_classifier.get_current_state()
            recording_status = self.video_recorder.get_recording_status()
            upload_stats = self.background_uploader.get_stats()
            frame_stats = self.frame_pool.get_efficiency_stats()
            camera_info = self.camera_manager.get_camera_info()
            
            return {
                # Core system state
                'auto_recording_enabled': self.auto_recording_enabled,
                'pose_detection_enabled': self.pose_detection_enabled,
                'current_pose_stage': self.current_pose_stage,
                'current_user_id': self.current_operator_id,  # 🔧 FIXED: Use correct variable
                
                # AI state
                'predicted_class': classifier_state.get('predicted_class', 'Unknown'),
                'p1_confidence': classifier_state.get('p1_confidence', 0.0),
                'p10_confidence': classifier_state.get('p10_confidence', 0.0),
                'models_loaded': classifier_state.get('model_loaded', False),
                
                # Recording state
                'is_recording': recording_status['is_recording'],
                'stop_recording_early': recording_status['stop_early_signal'],
                'video_path': recording_status.get('current_video_path'),
                'gcs_path': recording_status.get('gcs_path'),
                
                # Performance metrics
                'memory_efficiency': frame_stats['reuse_efficiency_percent'],
                'frames_in_pool': frame_stats['current_pool_size'],
                'upload_queue_size': upload_stats['queued'],
                'upload_success_rate': upload_stats['success_rate'],
                
                # System health
                'camera_initialized': camera_info['initialized'],
                'camera_mode': camera_info['current_mode'],
                'uploader_running': upload_stats['worker_running']
            }
            
        except Exception as e:
            log_error(logger, "GolfCameraSystem.get_system_status", e)
            return {'error': str(e)}
    
    def cleanup(self) -> None:
        """Clean up all system resources"""
        try:
            logger.info("🛑 Shutting down Golf Camera System...")
            
            # Stop any ongoing recording
            if self.video_recorder.is_recording:
                self.video_recorder.force_stop_recording()
            
            # Cleanup components
            self.pose_classifier.cleanup()
            self.camera_manager.cleanup()
            self.background_uploader.shutdown()
            
            log_success(logger, "Golf Camera System shutdown complete")
            
        except Exception as e:
            log_error(logger, "GolfCameraSystem.cleanup", e)

# Create global system instance
golf_system = GolfCameraSystem()

# === FLASK ROUTES ===

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        golf_system.camera_manager.generate_stream_frames(
            process_callback=golf_system.process_frame_for_streaming
        ),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Enhanced recording with assignee support"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        duration = int(data.get('duration', 10))
        user_id = data.get('user_id')  # Who is recording
        assignee_id = data.get('assignee_id')  # Who it's for

        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        
        # Set context if assignee provided
        if assignee_id:
            golf_system.set_user_context(user_id, data.get('role', 'coach'), assignee_id)
        else:
            golf_system.set_user_context(user_id, data.get('role', 'student'))
        
        success = golf_system.video_recorder.start_recording(
            duration=duration, 
            user_id=user_id,
            assignee_id=assignee_id
        )

        if success:
            return jsonify({
                "status": "success", 
                "message": f"Recording started - Operator: {user_id}, Video for: {assignee_id or user_id}"
            })
        else:
            return jsonify({"error": "Failed to start recording"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/set_user_context', methods=['POST'])
def set_user_context():
    """Enhanced user context setting"""
    try:
        data = request.get_json()
        if not data or 'user_id' not in data:
            return jsonify({"error": "user_id is required"}), 400
        
        user_id = data['user_id']
        role = data.get('role', 'student')
        assignee_id = data.get('assignee_id')  # Who the video is for
        
        success = golf_system.set_user_context(user_id, role, assignee_id)
        
        if success:
            return jsonify({
                "status": "success", 
                "operator_id": user_id,
                "assignee_id": golf_system.current_assignee_id,
                "role": role,
                "message": f"Recording as {role} - videos will go to: {golf_system.current_assignee_id}"
            })
        else:
            return jsonify({"error": "Failed to set user context"}), 500
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/toggle_auto_recording', methods=['POST'])
def toggle_auto_recording():
    """Toggle auto recording based on pose detection"""
    try:
        if not golf_system.pose_classifier.model_manager.is_loaded:
            return jsonify({"status": "error", "message": "Model not loaded yet"}), 400
        
        # 🔧 FIXED: Check if user context is set before enabling auto recording
        if not golf_system.current_operator_id:
            return jsonify({
                "status": "error", 
                "message": "No user context set. Please start Live Feed first to set user context."
            }), 400
        
        golf_system.auto_recording_enabled = not golf_system.auto_recording_enabled
        
        if not golf_system.auto_recording_enabled:
            golf_system.current_pose_stage = "waiting"
        
        return jsonify({
            "status": "success", 
            "auto_recording_enabled": golf_system.auto_recording_enabled,
            "message": f"Auto recording {'enabled' if golf_system.auto_recording_enabled else 'disabled'}",
            "user_id": golf_system.current_operator_id  # 🔧 FIXED: Use correct variable
        })
        
    except Exception as e:
        log_error(logger, "toggle_auto_recording endpoint", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/toggle_pose_detection', methods=['POST'])
def toggle_pose_detection():
    """Toggle pose detection"""
    try:
        golf_system.pose_detection_enabled = not golf_system.pose_detection_enabled
        
        return jsonify({
            "status": "success", 
            "pose_detection_enabled": golf_system.pose_detection_enabled,
            "message": f"Pose detection {'enabled' if golf_system.pose_detection_enabled else 'disabled'}"
        })
        
    except Exception as e:
        log_error(logger, "toggle_pose_detection endpoint", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/reload_models', methods=['POST'])
def reload_models():
    """Reload AI model from GCS"""
    try:
        force_download = request.json.get('force_download', False) if request.json else False
        
        success = golf_system.pose_classifier.reload_model(force_download=force_download)
        
        # ✅ CRITICAL: Re-disable transition validation after reload
        if success:
            golf_system.pose_classifier.validate_transitions = False
            print("🔧 RESTORED: Transition validation disabled after model reload")
        
        return jsonify({
            "status": "success" if success else "error", 
            "models_loaded": success,
            "message": "Model reloaded successfully" if success else "Failed to load model"
        })
        
    except Exception as e:
        log_error(logger, "reload_models endpoint", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/system_status')
def system_status():
    """Get comprehensive system status"""
    try:
        status = golf_system.get_system_status()
        return jsonify(status)
        
    except Exception as e:
        log_error(logger, "system_status endpoint", e)
        return jsonify({"status": "error", "message": str(e)}), 500

# Legacy endpoint for compatibility
@app.route('/recording_status')
def recording_status():
    """Get recording status (legacy compatibility)"""
    return system_status()

@app.route('/debug_model_info', methods=['GET'])
def debug_model_info():
    """Get detailed model information for debugging"""
    try:
        if not golf_system.pose_classifier.model_manager.is_loaded:
            return jsonify({"status": "error", "message": "Model not loaded"}), 400
        
        model_info = golf_system.pose_classifier.model_manager.get_model_info()
        test_results = golf_system.pose_classifier.model_manager.test_model_with_dummy_data()
        
        return jsonify({
            "model_info": model_info,
            "test_results": test_results
        })
        
    except Exception as e:
        log_error(logger, "debug_model_info endpoint", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/test_model_with_pose', methods=['POST'])
def test_model_with_pose():
    """Test model with current pose for debugging"""
    try:
        if not golf_system.pose_classifier.model_manager.is_loaded:
            return jsonify({"status": "error", "message": "Model not loaded"}), 400
        
        # Capture current frame
        frame = golf_system.camera_manager.capture_frame()
        if frame is None:
            return jsonify({"status": "error", "message": "Failed to capture frame"})
        
        # Test with current frame
        results = golf_system.pose_classifier.test_with_current_frame(frame)
        return jsonify(results)
        
    except Exception as e:
        log_error(logger, "test_model_with_pose endpoint", e)
        return jsonify({"status": "error", "message": str(e)}), 500

# Upload status routes
@app.route('/upload_status/<upload_id>')
def upload_status(upload_id):
    """Check status of specific upload"""
    status = golf_system.background_uploader.get_upload_info(upload_id)
    return jsonify(status)

@app.route('/recent_uploads')
def recent_uploads():
    """Get recent upload history"""
    recent = golf_system.background_uploader.get_recent_uploads()
    return jsonify(recent)

@app.route('/upload_queue_status')
def upload_queue_status():
    """Get upload queue information"""
    stats = golf_system.background_uploader.get_stats()
    return jsonify({
        'queue_size': stats['queued'],
        'recent_uploads': golf_system.background_uploader.get_recent_uploads(3),
        'stats': stats
    })

@app.route('/memory_stats')
def memory_stats():
    """Get memory efficiency statistics"""
    try:
        frame_stats = golf_system.frame_pool.get_efficiency_stats()
        
        # Optional: Add system memory info if psutil is available
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_stats = {
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available // (1024 * 1024),
                'memory_used_mb': memory.used // (1024 * 1024)
            }
        except ImportError:
            system_stats = {'note': 'Install psutil for system memory stats'}
        
        return jsonify({
            'frame_pool': frame_stats,
            'system_memory': system_stats
        })
        
    except Exception as e:
        log_error(logger, "memory_stats endpoint", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/system_debug')
def system_debug():
    """Get comprehensive system debug information"""
    try:
        debug_info = golf_system.pose_classifier.get_debug_info()
        camera_info = golf_system.camera_manager.get_camera_info()
        recorder_stats = golf_system.video_recorder.get_stats()
        uploader_stats = golf_system.background_uploader.get_stats()
        
        return jsonify({
            'ai_debug': debug_info,
            'camera_info': camera_info,
            'recorder_stats': recorder_stats,
            'uploader_stats': uploader_stats,
            'system_status': golf_system.get_system_status()
        })
        
    except Exception as e:
        log_error(logger, "system_debug endpoint", e)
        return jsonify({"status": "error", "message": str(e)}), 500

# Cleanup handler
@app.teardown_appcontext
def cleanup_system(error):
    """Cleanup system resources when app context tears down"""
    if error:
        log_error(logger, "Flask teardown", error)

def cleanup_on_exit():
    """Cleanup function to call on application exit"""
    golf_system.cleanup()

if __name__ == '__main__':
    import atexit
    
    # Register cleanup function
    atexit.register(cleanup_on_exit)
    
    # Print startup information
    print(f"📷 Starting Golf Swing Camera Server - FULLY MODULAR! 🚀")
    print(f"💾 Videos saved to: {golf_system.video_recorder.get_stats()['video_directory']}")
    print(f"🧠 AI model loaded: {golf_system.pose_classifier.model_manager.is_loaded}")
    print(f"🌐 Access interface at: http://<your-pi-ip>:{FLASK_PORT}")
    print(f"🎬 Preview: {golf_system.camera_manager.get_camera_info()['preview_size']}")
    print(f"🎬 Recording: {golf_system.camera_manager.get_camera_info()['recording_size']}")
    print(f"🚀 MODULAR: Clean separation of concerns for maintainability!")
    
    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        cleanup_on_exit()
    except Exception as e:
        log_error(logger, "Flask app run", e)
        cleanup_on_exit()
        raise