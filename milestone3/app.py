import csv
import datetime as dt
import io
import os
import queue
import threading
import time
from datetime import UTC, datetime, timedelta

import cv2
import jwt
import matplotlib

matplotlib.use('Agg')
import base64

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pyrebase
from flask import (
    Flask,
    Response,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'a-very-secret-key'
JWT_SECRET = 'another-very-secret-key'
JWT_ALGORITHM = 'HS256'

# --- Configuration & Initialization ---

config = {
    "apiKey": "AIzaSyBLHvHIg14s9yzhNlo6F_iDm2OFGrL5ZQ0",
    "authDomain": "crowd-count-fb60e.firebaseapp.com",
    "databaseURL": "https://crowd-count-fb60e-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "crowd-count-fb60e",
    "storageBucket": "crowd-count-fb60e.appspot.com",
    "messagingSenderId": "959425108511",
    "appId": "1:959425108511:web:da19cdffefb51f4bc6132a",
    "measurementId": "G-Q5VLQF7JCQ"
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# --- Safe Upload Folder ---
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# YOLO Model Setup
model = YOLO("yolov8n.pt")
person_class_id = 0
confidence_threshold = 0.4
iou_threshold = 0.7

# --- Global State (Per-user/per-area state using composite key) ---

USER_LOGS_REF = "user_logs"
USER_CAMERAS_REF = "user_cameras"
NEXT_ID = 1
STREAM_SOURCES = {}
FRAME_QUEUES = {}
ACTIVE_THREADS = {}
THREAD_RESULTS = {}
LIVE_COUNTS = {}
OBJECT_TRACKER = {}

# =======================================================
# --- AUTHENTICATION AND DATA UTILITY FUNCTIONS ---
# =======================================================

def generate_server_jwt(uid, email, username):
    payload = {
        "uid": uid, "email": email, "username": username,
        "iat": datetime.now(UTC),
        "exp": datetime.now(UTC) + timedelta(hours=1)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_server_jwt(token):
    if not token:
        return None
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        print("JWT expired.")
    except jwt.InvalidTokenError:
        print("Invalid JWT.")
    return None

def check_auth(request):
    return verify_server_jwt(request.cookies.get('server_jwt'))

def get_logged_in_uid(request):
    jwt_token = request.cookies.get('server_jwt')
    decoded = verify_server_jwt(jwt_token)
    return decoded.get('uid') if decoded else None

def get_auth_token(request):
    return session['user'].get('idToken') if 'user' in session else None

def is_user_admin(uid):
    return session['user'].get('is_admin', False)

def get_display_uid_and_token(request):
    logged_in_uid = get_logged_in_uid(request)
    if not logged_in_uid:
        return None, None, False

    logged_in_token = get_auth_token(request)

    is_admin_view = ('VIEWING_UID' in session) and (session.get('VIEWING_UID') != logged_in_uid)
    display_uid = session.get('VIEWING_UID') if is_admin_view else logged_in_uid

    return display_uid, logged_in_token, is_admin_view


# def get_effective_fetch_token(token, display_uid):
#     return token
def get_effective_fetch_token(token, display_uid):
    logged_in_uid = get_logged_in_uid(request)

    # If user is viewing their OWN dashboard → use their token
    if logged_in_uid == display_uid:
        return token

    # For both admin-view AND user-view:
    # We want read-only access to the selected user's data
    return None

def fetch_data_from_db(path, token=None):
    try:
        if token and isinstance(token, str):
            return db.child(path).get(token).val()
        return db.child(path).get().val()
    except Exception as e:
        print(f"DB fetch error: {e}")
        return None

def get_user_data(uid, id_token):
    path = f"user/{uid}"
    return fetch_data_from_db(path, id_token)

def get_all_users(id_token, current_user_uid, is_viewer_admin):
    users_data = fetch_data_from_db("user", id_token)
    if users_data is None:
        return []
    return [
        {'uid': uid, 'username': data.get('user_name', 'Unknown User')}
        for uid, data in users_data.items()
        if uid != current_user_uid and 'user_name' in data and data.get('role', '') == 'admin'
    ]



def get_logged_areas(display_uid, token):
    path = f"{USER_LOGS_REF}/{display_uid}"
    effective_token = get_effective_fetch_token(token, display_uid)
    all_logs = fetch_data_from_db(path, effective_token) or {}
    return sorted(all_logs.keys())

# =======================================================
# --- VIDEO PROCESSING & STREAM MANAGEMENT FUNCTIONS (Unchanged) ---
# =======================================================

def get_bbox_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2) 

def is_in_aoi_below_line(center_x, center_y, line_coords):
    x1, y1, x2, y2 = line_coords
    
    if x1 == x2: return False
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return center_y > m * center_x + b

def cleanup(uid, area_name):
    key = (uid, area_name)
    for d in [LIVE_COUNTS, FRAME_QUEUES, ACTIVE_THREADS, OBJECT_TRACKER, THREAD_RESULTS]: d.pop(key, None)
    if uid in STREAM_SOURCES and area_name in STREAM_SOURCES[uid]: STREAM_SOURCES[uid].pop(area_name)
    if uid in STREAM_SOURCES and not STREAM_SOURCES[uid]: STREAM_SOURCES.pop(uid)

def get_stream_info(uid, area_name):
    return STREAM_SOURCES.get(uid, {}).get(area_name)

def get_all_user_streams(uid):
    return STREAM_SOURCES.get(uid, {})

def start_stream_thread(uid, area_name, source_path, threshold, line_coords=None):
    global STREAM_SOURCES, FRAME_QUEUES, OBJECT_TRACKER
    STREAM_SOURCES.setdefault(uid, {})[area_name] = {'source': source_path, 'threshold': threshold, 'line_coords': line_coords, 'start_time': datetime.now().isoformat()}
    key = (uid, area_name)
    LIVE_COUNTS[key] = 0
    FRAME_QUEUES[key] = queue.Queue(maxsize=15)
    OBJECT_TRACKER[key] = {}
    thread = threading.Thread(target=continuous_video_analysis, args=(source_path, area_name, uid), daemon=True)
    thread.start()
    ACTIVE_THREADS[key] = thread

def continuous_video_analysis(source, area_name, owner_uid):
    global LIVE_COUNTS, FRAME_QUEUES, ACTIVE_THREADS, OBJECT_TRACKER, NEXT_ID
    key = (owner_uid, area_name)

    cap = cv2.VideoCapture(int(source) if source == "0" else source)
    if not cap.isOpened():
        print(f"Error: Cannot open source for {area_name} (User: {owner_uid})")
        cleanup(owner_uid, area_name)
        return

    frame_count = 0
    tracker = OBJECT_TRACKER.get(key, {})
    if not tracker:
        OBJECT_TRACKER[key] = {}
        tracker = OBJECT_TRACKER[key]

    try:
        while get_stream_info(owner_uid, area_name):
            ret, frame = cap.read()
            if frame is None or frame.size == 0:
                continue
            if not ret:
                print(f"Stream ended for {area_name} (User: {owner_uid})")
                break

            source_info = get_stream_info(owner_uid, area_name)
            if not source_info:
                break

            # 🔴 GET RECTANGLE AOI (if set by /set_aoi)
            rect = source_info.get("aoi_rect")

            results = model.predict(
                source=frame,
                classes=[person_class_id],
                conf=confidence_threshold,
                iou=iou_threshold,
                stream=False,
                verbose=False
            )

            result = next(iter(results), None)
            if not result or result.orig_img is None:
                time.sleep(0.01)
                continue

            processed_frame = result.orig_img.copy()
            person_count = 0
            live_count_in_aoi = 0
            new_tracker = {}
            active_ids = set()

            # 🟩 Process detections
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    center_x, center_y = get_bbox_center(x1, y1, x2, y2)

                    # Tracking match
                    match_id = -1
                    min_dist = float('inf')

                    for obj_id, obj_data in tracker.items():
                        prev_x, prev_y, _ = obj_data
                        dist = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                        if dist < min_dist and dist < 100:
                            min_dist = dist
                            match_id = obj_id

                    if match_id == -1:
                        match_id = NEXT_ID
                        NEXT_ID += 1
                        is_inside = False
                    else:
                        _, _, is_inside = tracker.get(match_id, (center_x, center_y, False))

                    person_count += 1

                    # 🟢 RECTANGLE AOI CHECK
                    inside = False
                    if rect:
                        rx1, ry1, rx2, ry2 = rect
                        inside = (rx1 <= center_x <= rx2) and (ry1 <= center_y <= ry2)

                    # Coloring based on AOI
                    if rect:
                        if inside:
                            live_count_in_aoi += 1
                            color = (0, 255, 0)     # green: inside AOI
                        else:
                            color = (255, 0, 255)   # purple: outside AOI
                    else:
                        color = (255, 0, 0)         # red: no AOI set

                    # Update tracker
                    new_tracker[match_id] = (center_x, center_y, inside)
                    active_ids.add(match_id)

                    # Draw person box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(processed_frame, str(match_id),
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                tracker.clear()
                tracker.update(new_tracker)

            # 🟦 DRAW RECTANGLE AOI + UPDATE LIVE COUNTS
            if rect:
                rx1, ry1, rx2, ry2 = rect

                cv2.rectangle(processed_frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3)

                cv2.putText(
                    processed_frame,
                    f"IN RESTRICTED ZONE: {live_count_in_aoi}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                LIVE_COUNTS[key] = live_count_in_aoi
            else:
                LIVE_COUNTS[key] = person_count
                cv2.putText(
                    processed_frame,
                    f"TOTAL COUNT: {person_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3
                )

            # Push frame to queue
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            if key in FRAME_QUEUES:
                try:
                    FRAME_QUEUES[key].put_nowait(frame_bytes)
                except queue.Full:
                    pass

            time.sleep(0.01)
            frame_count += 1

    finally:
        cap.release()
        cleanup(owner_uid, area_name)
        THREAD_RESULTS[key] = f"Analysis ended. Frames processed: {frame_count}"


def video_feed_generator(uid, area_name):
    key = (uid, area_name)
    if key not in FRAME_QUEUES: return
    q = FRAME_QUEUES[key]
    while get_stream_info(uid, area_name):
        try:
            frame_bytes = q.get(timeout=0.1)
            yield (b'--frame_boundary\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            q.task_done()
        except queue.Empty: continue
        except Exception as e: print(f"Video feed error for {area_name} (User: {uid}): {e}"); break

# =======================================================
# --- CHART GENERATION FUNCTIONS (Unchanged) ---
# =======================================================


def generate_crowd_chart_image():
    """Generates a comparison (grouped bar) chart for crowd counts over time."""
    display_uid, token, _ = get_display_uid_and_token(request)
    if not display_uid:
        return ""

    log_entries = []
    effective_token = get_effective_fetch_token(token, display_uid)
    all_logs = fetch_data_from_db(f"{USER_LOGS_REF}/{display_uid}", effective_token) or {}
    log_entries = [entry for area in all_logs.values() if isinstance(area, dict) for entry in area.values()]

    if not log_entries:
        return ""

    # Sort and limit
    log_entries.sort(key=lambda x: x.get('time', ''))
    limited_log = log_entries[-10:]  # last 10 points for clarity
    data_by_area = {}

    for entry in limited_log:
        try:
            t = datetime.fromisoformat(entry['time'].replace('Z', '+00:00'))
            area = entry['area']
            data_by_area.setdefault(area, {'times': [], 'counts': []})
            data_by_area[area]['times'].append(t)
            data_by_area[area]['counts'].append(entry['count'])
        except Exception:
            continue

    if not data_by_area:
        return ""

    # Prepare common x-axis (timestamps)
    all_times = sorted({t for data in data_by_area.values() for t in data['times']})
    x = np.arange(len(all_times))
    width = 0.8 / len(data_by_area)  # space bubbles for each area

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (area, data) in enumerate(data_by_area.items()):
        y = []
        for t in all_times:
            if t in data['times']:
                y.append(data['counts'][data['times'].index(t)])
            else:
                y.append(0)
        ax.bar(x + i * width, y, width, label=area)

    ax.set_xticks(x + width * (len(data_by_area) - 1) / 2)
    ax.set_xticklabels([t.strftime('%H:%M:%S') for t in all_times], rotation=45, ha='right')
    ax.set_ylabel('Number of People')
    ax.set_title('Crowd Count Comparison Across Areas')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getbuffer()).decode("ascii")



def generate_density_chart_image():
    """Generates a live density donut chart for active streams (user-specific)."""
    display_uid, token, _ = get_display_uid_and_token(request)
    if not display_uid:
        return ""

    user_streams = get_all_user_streams(display_uid)

    areas, densities, colors = [], [], []
    for area_name, info in user_streams.items():
        key = (display_uid, area_name)
        count = LIVE_COUNTS.get(key)
        
        if info and isinstance(count, int) and (threshold := info.get('threshold')) and threshold > 0:
            density = min((count / threshold) * 100, 100)
            areas.append(area_name)
            densities.append(density)
            colors.append('green' if density < 50 else 'gold' if density < 90 else 'red')

    if not areas:
        return ""

    fig, ax = plt.subplots(figsize=(7, 7))

    # Pie chart (donut)
    wedges, texts, autotexts = ax.pie(
        densities,
        labels=areas,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10, 'color': 'black'},
        wedgeprops=dict(width=0.35, edgecolor='white')  # this makes it a donut
    )

    # Add center text
    total_density = np.mean(densities)
    ax.text(0, 0, f'Avg\n{total_density:.1f}%', ha='center', va='center',
            fontsize=13, fontweight='bold')

    ax.set_title(f'Live Density Donut Chart for {display_uid}', fontsize=14, pad=20)

    plt.tight_layout()

    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getbuffer()).decode("ascii")



# =======================================================
# --- FLASK ROUTES (Modified: login, register, profile, update_profile, home, restricted pages) ---
# =======================================================

@app.route('/')
def root():
    return redirect(url_for('signin'))

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/login', methods=['POST'])
def login():
    try:
        email, password = request.form['email'], request.form['password']
        user = auth.sign_in_with_email_and_password(email, password)
        uid, id_token = user['localId'], user['idToken']
        
        user_data = get_user_data(uid, id_token)
        if user_data is None:
            print(f"Warning: User {uid} signed in but profile data is missing. Logging out.")
            return render_template('signin.html', Error="Profile data error. Please register again.")
            
        username = user_data.get('user_name', 'User')
        user_role = user_data.get('role', 'user')
        
        # --- ADMIN ROLE CHECK ---
        is_admin_user = (user_role == 'admin')
        # ------------------------
        
        server_jwt = generate_server_jwt(uid, email, username)
        
        # Store is_admin status in session 
        session['user'] = {'uid': uid, 'idToken': id_token, 'jwt': server_jwt, 'is_admin': is_admin_user}
        
        session.pop('VIEWING_UID', None)
        response = make_response(redirect(url_for('index')))
        response.set_cookie('server_jwt', server_jwt, httponly=True, samesite='Lax')
        return response
    except Exception as e:
        print(e)
        return render_template('signin.html', Error="Invalid username/password!!")

@app.route('/register', methods=['POST'])
def register():
    try:
        email, password, username = request.form['email'], request.form['password'], request.form['username']
        # Default role is 'user'. Role is received from the form.
        role = request.form.get('role', 'user') 
        
        user = auth.create_user_with_email_and_password(email, password)
        uid, id_token = user['localId'], user['idToken']
        
        db.child("user").child(uid).set({
            "user_name": username, 
            "role": role 
        }, id_token)
        
        return redirect(url_for('signin'))
    except Exception as e:
        print(e)
        return render_template('signin.html', Error="Could not register. Email may already be in use.", mode='register')

@app.route('/index')
def index():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))

    _, _, is_admin_view = get_display_uid_and_token(request)
    logged_in_uid = decoded.get('uid')
    is_admin = is_user_admin(logged_in_uid)

    return render_template(
        'index.html',
        username=decoded.get('username', 'User'),
        is_admin=is_admin,
        is_admin_view=is_admin_view
    )

@app.context_processor
def inject_user_role():
    """Make `is_admin` available in all templates globally."""
    decoded = check_auth(request)
    is_admin = False
    if decoded:
        uid = decoded.get('uid')
        is_admin = is_user_admin(uid)
    return dict(is_admin=is_admin)

@app.route('/view_user/<target_uid>')
def view_user(target_uid):
    """
    Allows a user to initiate the view of another user's dashboard.
    """
    logged_in_uid = get_logged_in_uid(request)
    
    if not logged_in_uid:
        return redirect(url_for('signin'))
    
    if logged_in_uid != target_uid:
        session['VIEWING_UID'] = target_uid
    else:
        # Clear admin view if they click on their own profile
        session.pop('VIEWING_UID', None) 
        
    return redirect(url_for('home'))

@app.route('/exit_user_view')
def exit_user_view():
    session.pop('VIEWING_UID', None)
    return redirect(url_for('home'))


@app.route('/home')
def home():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))

    logged_in_uid = decoded.get('uid')
    token = get_auth_token(request)
    display_uid, _, is_admin_view = get_display_uid_and_token(request)
    is_admin = is_user_admin(logged_in_uid)

    # --- Determine the token to use for data fetching ---
    effective_token = token
    # ----------------------------------------------------

    # Get the displayed username (if admin is viewing someone else's dashboard)
    displayed_username = (
        get_user_data(display_uid, effective_token).get('user_name', 'Viewed User')
        if is_admin_view else decoded.get('username', 'User')
    )

    # Get user list based on the viewer's role
    #if not is_admin_view:
    all_users = get_all_users(token, logged_in_uid, is_admin)
    # else:
    #     all_users = []

    # === EXISTING CAMERA & CHARTS ===
    current_user_streams = STREAM_SOURCES.get(display_uid, {})
    historical_chart_data = generate_crowd_chart_image()
    density_chart_data = generate_density_chart_image()

    camera_data = [
        {
            'areaName': name,
            'sourceType': 'Webcam' if info['source'] == '0' else 'Video File',
            'capacityThreshold': info.get('threshold', 'N/A')
        }
        for name, info in current_user_streams.items()
    ]

    # === 🟢 NEW: Calculate Live Stats for Dashboard ===
    total_cameras = len(current_user_streams)

    live_counts = []
    for area_name, info in current_user_streams.items():
        key = (display_uid, area_name)
        count = LIVE_COUNTS.get(key)
        if isinstance(count, int):
            live_counts.append(count)

    avg_count = sum(live_counts) / len(live_counts) if live_counts else 0

    peak_density = 0
    for area_name, info in current_user_streams.items():
        key = (display_uid, area_name)
        count = LIVE_COUNTS.get(key)
        if info and isinstance(count, int) and (threshold := info.get('threshold')) and threshold > 0:
            density = min((count / threshold) * 100, 100)
            peak_density = max(peak_density, density)

    # === Render Dashboard Template ===
    return render_template(
        'home.html',
        cameras=camera_data,
        historical_chart_data=historical_chart_data,
        density_chart_data=density_chart_data,
        all_users=all_users,
        is_admin_view=is_admin_view,
        displayed_username=displayed_username,
        total_cameras=total_cameras,
        avg_count=int(avg_count),
        peak_density=int(peak_density)
    )

@app.route('/home_stats')
def home_stats():
    decoded = check_auth(request)
    if not decoded:
        return jsonify({"error": "Unauthorized"}), 401

    display_uid, _, _ = get_display_uid_and_token(request)
    current_user_streams = STREAM_SOURCES.get(display_uid, {})

    total_cameras = len(current_user_streams)
    live_counts = [LIVE_COUNTS.get((display_uid, a), 0) for a in current_user_streams]
    avg_count = sum(live_counts) / len(live_counts) if live_counts else 0

    peak_density = 0
    for a, info in current_user_streams.items():
        count = LIVE_COUNTS.get((display_uid, a))
        if info and isinstance(count, int) and (th := info.get('threshold')) and th > 0:
            density = min((count / th) * 100, 100)
            peak_density = max(peak_density, density)

    return jsonify({
        "total_cameras": total_cameras,
        "avg_count": int(avg_count),
        "peak_density": int(peak_density)
    })



# 🔑 RESTRICTED PAGES FOR STANDARD USERS (Standard Users redirect to home)

@app.route('/vid_analy')
def vid_analy():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))
    
    logged_in_uid = decoded.get('uid')
    # Only admins can access this page
    if not is_user_admin(logged_in_uid):
        return redirect(url_for('home'))
        
    # Prevent action if currently viewing another user's dashboard (admin view mode)
    if get_display_uid_and_token(request)[2]:
        return redirect(url_for('home'))

    # Fetch current streams
    current_user_streams = get_all_user_streams(logged_in_uid)
    camera_data = [
        {
            'areaName': name,
            'sourceType': 'Webcam' if info['source'] == '0' else 'Video File'
        }
        for name, info in current_user_streams.items()
    ]
    return render_template("vid_analy.html", cameras=camera_data)

@app.route('/cam_manage', methods=['GET'])
def cam_manage_page():
    if check_auth(request) is None:
        return redirect(url_for('signin'))
    
    logged_in_uid = get_logged_in_uid(request)
        
    # Redirect if the user is a standard (non-admin) user
    if not is_user_admin(logged_in_uid):
        return redirect(url_for('home'))
        
    # Block admin view mode from managing other users’ cameras
    if get_display_uid_and_token(request)[2]:
        return redirect(url_for('home'))
        
    return render_template("cam_manage.html")

@app.route('/define_aoi/<area_name>')
def define_aoi_page(area_name):
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))

    logged_in_uid = decoded.get('uid')
    if not is_user_admin(logged_in_uid):
        return redirect(url_for('home'))

    # Ensure the stream exists for the given area
    if not get_stream_info(logged_in_uid, area_name):
        return "No active stream for this area.", 404

    return render_template('define_aoi.html', area_name=area_name)

@app.route('/set_aoi', methods=['POST'])
def set_aoi():
    logged_in_uid = get_logged_in_uid(request)
    if not logged_in_uid:
        return jsonify({"message": "Unauthorized."}), 401

    # Only Admins can create AOI
    if not is_user_admin(logged_in_uid):
        return jsonify({"error": "Access Denied"}), 403

    data = request.get_json()

    area_name = data.get("areaName")
    rect = data.get("rect_coords")   # NEW rectangle AOI

    # Validate rectangle AOI
    if not area_name or not rect:
        return jsonify({"message": "Invalid AOI rectangle."}), 400

    if not get_stream_info(logged_in_uid, area_name):
        return jsonify({"message": "Stream inactive."}), 404

    # Save the rectangle AOI into STREAM_SOURCES
    STREAM_SOURCES[logged_in_uid][area_name]["aoi_rect"] = [
        rect["x1"], rect["y1"], rect["x2"], rect["y2"]
    ]

    return jsonify({"message": "AOI rectangle saved"}), 200


# ----------------------------------------------------------------------

@app.route('/video_feed/<area_name>')
def video_feed(area_name):
    display_uid, _, _ = get_display_uid_and_token(request)
    if not display_uid: return "Unauthorized.", 401

    if not get_stream_info(display_uid, area_name):
        return "No active stream.", 404

    return Response(video_feed_generator(display_uid, area_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame_boundary')


@app.before_request
def check_admin_view_permissions():
    """Prevents modification actions when viewing another user's data."""
    if request.endpoint in ['login', 'register', 'signin', 'signout']:
        return

    # Block all modification actions in admin-view mode
    if request.method == 'POST' or request.endpoint in [
        'cam_manage_page', 'vid_analy', 'define_aoi_page', 
        'upload_video', 'upload_cam', 'set_aoi', 
        'stop_stream', 'delete_history', 'api_camera_config'
    ]:
        _, _, is_admin_view = get_display_uid_and_token(request)
        if is_admin_view:
            if request.is_json:
                return jsonify({"message": "Restricted in Admin View."}), 403
            return redirect(url_for('home'))


@app.route('/get_total_count', methods=['GET'])
def get_total_count():
    display_uid, _, _ = get_display_uid_and_token(request)
    if not display_uid: return jsonify({"totalCount": 0, "activeStreams": 0})

    user_streams = get_all_user_streams(display_uid)
    total_count = sum(LIVE_COUNTS.get((display_uid, name), 0) for name in user_streams.keys())

    return jsonify({"totalCount": total_count, "activeStreams": len(user_streams)})

@app.route('/get_count/<area_name>', methods=['GET'])
def get_count(area_name):
    display_uid, _, _ = get_display_uid_and_token(request)
    if not display_uid: return jsonify({"areaName": area_name, "count": "N/A", "isActive": False, "threshold": "N/A"})

    info = get_stream_info(display_uid, area_name)
    key = (display_uid, area_name)

    return jsonify({
        "areaName": area_name,
        "count": LIVE_COUNTS.get(key, "N/A"),
        "isActive": info is not None,
        "threshold": info.get('threshold', 'N/A') if info else 'N/A'
    })

@app.route('/upload_video', methods=['POST'])
def upload_video():
    logged_in_uid = get_logged_in_uid(request)
    if not logged_in_uid: return jsonify({"error": "Unauthorized"}), 401
    
    # Only Admins can use this page
    if not is_user_admin(logged_in_uid): return jsonify({"error": "Access Denied"}), 403

    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400
    video_file = request.files["video"]
    area_name = request.form.get("areaName", "Unknown")
    threshold = int(request.form.get("capacityThreshold", 0))

    if get_stream_info(logged_in_uid, area_name):
        return jsonify({"message": "Stream active. Stop first."}), 409

    path = os.path.join(UPLOAD_FOLDER, secure_filename(video_file.filename))
    video_file.save(path)
    if not os.path.exists(path):
        return jsonify({"error": "Save failed"}), 500

    start_stream_thread(logged_in_uid, area_name, path, threshold)
    return jsonify({"message": f"Processing started for {area_name}."}), 200

@app.route('/upload_cam', methods=['POST'])
def upload_cam():
    logged_in_uid = get_logged_in_uid(request)
    if not logged_in_uid: return jsonify({"error": "Unauthorized"}), 401

    # Only Admins can use this page
    if not is_user_admin(logged_in_uid): return jsonify({"error": "Access Denied"}), 403

    area_name = request.form.get("areaName", "Unknown")
    threshold = int(request.form.get("capacityThreshold", 0))

    if get_stream_info(logged_in_uid, area_name):
        return jsonify({"message": "Stream active."}), 409

    start_stream_thread(logged_in_uid, area_name, "0", threshold)
    return jsonify({"message": f"Camera started for {area_name}."}), 200



@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    logged_in_uid = get_logged_in_uid(request)
    if not logged_in_uid: return jsonify({"message": "Unauthorized."}), 401

    # Only Admins can use this page
    if not is_user_admin(logged_in_uid): return jsonify({"error": "Access Denied"}), 403

    area_name = request.json.get("areaName")

    if get_stream_info(logged_in_uid, area_name):
        cleanup(logged_in_uid, area_name) 
        return jsonify({"message": "Stream stopped."}), 200

    return jsonify({"message": "No stream."}), 200

@app.route('/signout')
def signout():
    logged_in_uid = get_logged_in_uid(request)
    if logged_in_uid in STREAM_SOURCES:
        for area in list(STREAM_SOURCES[logged_in_uid].keys()):
            cleanup(logged_in_uid, area)

    cv2.destroyAllWindows()
    session.pop('VIEWING_UID', None)
    session.pop('user', None)
    response = make_response(redirect(url_for('signin')))
    response.set_cookie('server_jwt', '', expires=0)
    return response


@app.route('/log_area_count', methods=['POST'])
def log_area_count():
    logged_in_uid = get_logged_in_uid(request)
    token = get_auth_token(request)
    if not logged_in_uid:
        return jsonify({"message": "Unauthorized."}), 401

    data = request.get_json()
    if not all(k in data for k in ('area', 'count', 'time')):
        return jsonify({"message": "Missing data."}), 400

    entry = {'area': data['area'], 'count': int(data['count']), 'time': data['time']}
    try:
        db.child(USER_LOGS_REF).child(logged_in_uid).child(data['area']).push(entry, token)
        return jsonify({"message": "Logged.", "area": data['area']}), 200
    except Exception as e:
        print(f"Firebase Log error: {e}")
        return jsonify({"message": "DB write failed."}), 500
    
@app.route('/get_historical_log', methods=['GET'])
def get_historical_log():
    display_uid, token, _ = get_display_uid_and_token(request)
    if not display_uid:
        return jsonify({"message": "Unauthorized."}), 410
    
    effective_token = get_effective_fetch_token(token, display_uid)
    # Fetch all logs for the current (or viewed) user
    all_logs = fetch_data_from_db(f"{USER_LOGS_REF}/{display_uid}", effective_token) or {}
    logs = [entry for area in all_logs.values() if isinstance(area, dict) for entry in area.values()]

    if not logs:
        return jsonify({"message": "No data."}), 204

    # Filter recent logs (last 5 minutes)
    five_min_ago = datetime.now(UTC) - timedelta(minutes=5)
    recent = []
    for e in logs:
        try:
            if datetime.fromisoformat(e['time'].replace('Z', '+00:00')) > five_min_ago:
                recent.append(e)
        except ValueError:
            continue

    return jsonify(recent)


@app.route('/get_server_chart_data', methods=['GET'])
def get_server_chart_data():
    h = generate_crowd_chart_image()
    d = generate_density_chart_image()
    return jsonify({"historical_chart_data": h, "density_chart_data": d}) if h or d else (jsonify({"message": "No data."}), 204)

@app.route('/api/camera_config', methods=['GET', 'POST'])
def api_camera_config():
    logged_in_uid = get_logged_in_uid(request)
    token = get_auth_token(request)

    if not logged_in_uid or not token:
        if request.method == 'GET':
            return jsonify({"configs": []}), 200
        return jsonify({"message": "Unauthorized."}), 401

    # Only admins can access camera configuration
    if not is_user_admin(logged_in_uid):
        return jsonify({"error": "Access Denied"}), 403

    # === GET REQUEST ===
    if request.method == 'GET':
        data = fetch_data_from_db(f"{USER_CAMERAS_REF}/{logged_in_uid}", token) or {}
        configs = list(data.values()) if isinstance(data, dict) else []
        return jsonify({"configs": configs}), 200

    # === POST REQUEST ===
    try:
        payload = request.get_json(silent=True) or []
        if not isinstance(payload, list):
            return jsonify({"message": "Invalid payload"}), 400

        to_store = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            cam_id = item.get("id") or item.get("areaName", f"cam_{int(time.time() * 1000)}")
            to_store[cam_id] = item

        db.child(USER_CAMERAS_REF).child(logged_in_uid).set(to_store, token)
        return jsonify({"message": "Saved", "count": len(to_store)}), 200

    except Exception as e:
        print(f"Camera config error: {e}")
        return jsonify({"message": "DB error."}), 500


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
