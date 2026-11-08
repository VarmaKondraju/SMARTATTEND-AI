import os, csv, time, sqlite3, secrets, shutil, threading, sys, base64
from io import StringIO
from pathlib import Path
from datetime import datetime, date
from functools import wraps

import cv2
import numpy as np
from flask import (Flask, Response, abort, flash, redirect, render_template,
                   request, session, url_for, make_response, jsonify)
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
FACES_DIR = DATA_DIR / 'faces'
DB_PATH = Path(os.environ.get('SMARTATTEND_DB', DATA_DIR / 'smartattend.db'))
DATA_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('SMARTATTEND_SECRET') or secrets.token_hex(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')

camera = None
camera_lock = threading.RLock()
recognizer = None
recognizer_ready = False
recognizer_label_to_id = {}
from pathlib import Path

CASCADE_PATH = Path(__file__).resolve().parent / "data" / "cascades" / "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(str(CASCADE_PATH))

if FACE_CASCADE.empty():
    raise RuntimeError(f"Face detector could not load: {CASCADE_PATH}")


def connect_db():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def init_db():
    """Create only new auth/audit tables. Existing students/attendance data is retained."""
    with connect_db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT NOT NULL,
            session_date TEXT NOT NULL, marked_at TEXT NOT NULL,
            UNIQUE(student_id, session_date),
            FOREIGN KEY(student_id) REFERENCES students(student_id))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL, password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin','faculty')),
            active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
            username TEXT NOT NULL, action TEXT NOT NULL, details TEXT,
            created_at TEXT NOT NULL)''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(session_date)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at)')


def audit(action, details=''):
    username = session.get('username', 'system')
    uid = session.get('user_id')
    with connect_db() as conn:
        conn.execute('INSERT INTO audit_logs(user_id, username, action, details, created_at) VALUES(?,?,?,?,?)',
                     (uid, username, action, details[:500], datetime.now().isoformat(timespec='seconds')))


def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login', next=request.path))
        return fn(*args, **kwargs)
    return wrapped


def roles_required(*roles):
    def deco(fn):
        @wraps(fn)
        @login_required
        def wrapped(*args, **kwargs):
            if session.get('role') not in roles:
                abort(403)
            return fn(*args, **kwargs)
        return wrapped
    return deco


@app.before_request
def csrf_guard():
    if request.method == 'POST' and request.endpoint not in ('setup', 'login'):
        token = session.get('csrf_token')
        submitted_token = request.headers.get('X-CSRF-Token') if request.is_json else request.form.get('_csrf')
        if not token or submitted_token != token:
            abort(400, 'Security token missing or expired. Refresh the page and try again.')

@app.context_processor
def inject_globals():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_urlsafe(24)
    return {'csrf_token': session['csrf_token'], 'current_user': session.get('full_name'),
            'current_role': session.get('role')}


def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            if os.name == 'nt':
                camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                backend = getattr(cv2, 'CAP_AVFOUNDATION', cv2.CAP_ANY) if sys.platform == 'darwin' else cv2.CAP_ANY
                camera = cv2.VideoCapture(0, backend)
            if not camera.isOpened():
                camera = cv2.VideoCapture(0)
        return camera


def train_recognizer():
    global recognizer, recognizer_ready, recognizer_label_to_id
    if not hasattr(cv2, 'face') or not hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
        recognizer_ready = False
        return False
    images, labels, mapping = [], [], {}
    with connect_db() as conn:
        rows = conn.execute('SELECT student_id FROM students ORDER BY student_id').fetchall()
    for label, row in enumerate(rows):
        sid = row['student_id']; mapping[label] = sid
        folder = FACES_DIR / sid
        if not folder.is_dir():
            continue
        for path in folder.glob('*.jpg'):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            try:
                found = FACE_CASCADE.detectMultiScale(image, 1.1, 4, minSize=(45, 45))
            except cv2.error:
                found = ()
            if len(found):
                x, y, w, h = max(found, key=lambda box: box[2] * box[3])
                image = image[y:y+h, x:x+w]
            images.append(cv2.resize(image, (160, 160))); labels.append(label)
    if not images:
        recognizer_ready = False; recognizer = None; recognizer_label_to_id = mapping
        return False
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.asarray(labels, dtype=np.int32))
    recognizer_label_to_id = mapping; recognizer_ready = True
    return True


def mark_attendance(student_id):
    now = datetime.now()
    with connect_db() as conn:
        cur = conn.execute('INSERT OR IGNORE INTO attendance(student_id, session_date, marked_at) VALUES(?,?,?)',
                           (student_id, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')))
        return cur.rowcount == 1


def camera_frames():
    global recognizer_ready
    while True:
        cap = get_camera()
        with camera_lock:
            ok, frame = cap.read()
        if not ok:
            frame = np.zeros((480, 800, 3), dtype=np.uint8)
            cv2.putText(frame, 'Camera unavailable - check browser/OS permissions', (22, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (240, 240, 240), 2)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                faces = FACE_CASCADE.detectMultiScale(gray, 1.15, 5, minSize=(60, 60)) if not FACE_CASCADE.empty() else ()
            except cv2.error:
                faces = ()
            if recognizer is None:
                train_recognizer()
            for (x, y, w, h) in faces:
                label_text, color = 'Unknown · not marked', (0, 165, 255)
                if recognizer is not None and recognizer_ready:
                    try:
                        label, confidence = recognizer.predict(cv2.resize(gray[y:y+h, x:x+w], (160, 160)))
                        sid = recognizer_label_to_id.get(label)
                        if sid and confidence < 65:
                            with connect_db() as conn:
                                student = conn.execute('SELECT name FROM students WHERE student_id=?', (sid,)).fetchone()
                            if student:
                                label_text, color = f"{student['name']} · {sid}", (70, 205, 130)
                                mark_attendance(sid)
                    except (cv2.error, sqlite3.Error):
                        pass
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label_text[:48], (x, max(22, y-10)), cv2.FONT_HERSHEY_SIMPLEX, .52, color, 2)
        ok, encoded = cv2.imencode('.jpg', frame)
        if ok:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded.tobytes() + b'\r\n'
        time.sleep(.06)


def attendance_rows(start=None, end=None, search=''):
    start = start or date.today().isoformat(); end = end or start
    sql = '''SELECT s.student_id, s.name, a.session_date, a.marked_at,
             CASE WHEN a.id IS NULL THEN 'Absent' ELSE 'Present' END status
             FROM students s LEFT JOIN attendance a ON a.student_id=s.student_id
             AND a.session_date BETWEEN ? AND ? WHERE (s.student_id LIKE ? OR s.name LIKE ?)
             ORDER BY a.session_date DESC, s.name COLLATE NOCASE'''
    with connect_db() as conn:
        return conn.execute(sql, (start, end, f'%{search}%', f'%{search}%')).fetchall()


@app.route('/setup', methods=['GET', 'POST'])
def setup():
    with connect_db() as conn:
        count = conn.execute('SELECT COUNT(*) n FROM users').fetchone()['n']
    if count:
        return redirect(url_for('login'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        name = request.form.get('full_name', '').strip()
        password = request.form.get('password', '')
        if not username or not name or len(password) < 8:
            flash('Enter all fields. Password must contain at least 8 characters.', 'error')
        else:
            with connect_db() as conn:
                conn.execute('INSERT INTO users(username,full_name,password_hash,role,created_at) VALUES(?,?,?,?,?)',
                             (username, name, generate_password_hash(password, method='pbkdf2:sha256:600000'), 'admin', datetime.now().isoformat(timespec='seconds')))
            flash('Administrator account created. Sign in to continue.', 'success')
            return redirect(url_for('login'))
    return render_template('setup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    with connect_db() as conn:
        count = conn.execute('SELECT COUNT(*) n FROM users').fetchone()['n']
    if not count:
        return redirect(url_for('setup'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        with connect_db() as conn:
            user = conn.execute('SELECT * FROM users WHERE username=? AND active=1', (username,)).fetchone()
        if user and check_password_hash(user['password_hash'], request.form.get('password', '')):
            session.clear(); session['user_id'] = user['id']; session['username'] = user['username']
            session['full_name'] = user['full_name']; session['role'] = user['role']; session['csrf_token'] = secrets.token_urlsafe(24)
            audit('LOGIN', 'Successful sign in')
            return redirect(url_for('dashboard'))
        flash('Incorrect username or password.', 'error')
    return render_template('login.html')


@app.route('/logout', methods=['POST'])
@login_required
def logout():
    audit('LOGOUT', 'Signed out'); session.clear(); return redirect(url_for('login'))


@app.route('/')
@roles_required('admin', 'faculty')
def dashboard():
    today = date.today().isoformat()
    with connect_db() as conn:
        total = conn.execute('SELECT COUNT(*) n FROM students').fetchone()['n']
        present = conn.execute('SELECT COUNT(*) n FROM attendance WHERE session_date=?', (today,)).fetchone()['n']
        recent = conn.execute('''SELECT a.student_id,s.name,a.marked_at FROM attendance a
             JOIN students s ON s.student_id=a.student_id WHERE a.session_date=? ORDER BY a.marked_at DESC LIMIT 7''', (today,)).fetchall()
    return render_template('dashboard.html', total=total, present=present, absent=max(0,total-present), recent=recent, model_ready=recognizer_ready)


@app.route('/students')
@roles_required('admin', 'faculty')
def students():
    q = request.args.get('q', '').strip()
    with connect_db() as conn:
        rows = conn.execute('SELECT * FROM students WHERE student_id LIKE ? OR name LIKE ? ORDER BY name COLLATE NOCASE', (f'%{q}%', f'%{q}%')).fetchall()
    return render_template('students.html', students=rows, q=q)


@app.route('/enroll', methods=['GET', 'POST'])
@roles_required('admin', 'faculty')
def enroll():
    # Enrollment is performed from the browser's own camera stream. This avoids
    # opening a second macOS camera session in OpenCV while the preview is active.
    if request.method == 'POST':
        flash('Use the Enroll Student button so the browser can capture from the live preview.', 'error')
        return redirect(url_for('enroll'))
    return render_template('enroll.html')


@app.route('/enroll/browser-capture', methods=['POST'])
@roles_required('admin', 'faculty')
def enroll_browser_capture():
    payload = request.get_json(silent=True) or {}
    sid = str(payload.get('student_id', '')).strip()
    name = str(payload.get('name', '')).strip()
    frames = payload.get('frames') or []
    if not sid or not name or len(sid) > 40 or len(name) > 100 or any(x in sid for x in ('/', '\\', '..')):
        return jsonify(ok=False, message='Enter a valid student ID and full name.'), 400
    if not isinstance(frames, list) or len(frames) < 3:
        return jsonify(ok=False, message='Could not capture enough camera frames. Please try again.'), 400

    folder = FACES_DIR / sid
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    for item in frames[:5]:
        try:
            encoded = item.split(',', 1)[1] if ',' in item else item
            raw = base64.b64decode(encoded, validate=True)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                continue
            # Browser camera frame is already centered by the on-screen guide.
            # Save a centered portrait crop; Haar detection is not a prerequisite
            # for enrollment, so a detector miss cannot block a student signup.
            h, w = frame.shape[:2]
            side = max(80, min(int(h * 0.92), int(w * 0.72)))
            cx, cy = w // 2, int(h * 0.50)
            x1 = max(0, min(w - side, cx - side // 2))
            y1 = max(0, min(h - side, cy - side // 2))
            crop = frame[y1:y1+side, x1:x1+side]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
            target = folder / f'{time.time_ns()}_{saved}.jpg'
            if cv2.imwrite(str(target), gray):
                saved += 1
        except Exception as exc:
            app.logger.warning('Could not save browser face frame for %s: %s', sid, exc)

    if saved < 3:
        return jsonify(ok=False, message=f'Only {saved} usable photos were received. Check camera permission and try again.'), 400
    with connect_db() as conn:
        conn.execute('INSERT INTO students(student_id,name,created_at) VALUES(?,?,?) ON CONFLICT(student_id) DO UPDATE SET name=excluded.name',
                     (sid, name, datetime.now().isoformat(timespec='seconds')))
    train_recognizer()
    audit('STUDENT_ENROLLED', f'{sid} · {name} · {saved} browser samples')
    return jsonify(ok=True, message=f'{name} enrolled successfully with {saved} face samples.')


@app.route('/kiosk')
def kiosk():
    return render_template('kiosk.html')

@app.route('/camera')
def camera_page():
    return Response(camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/attendance')
@roles_required('admin', 'faculty')
def attendance():
    selected = request.args.get('date', date.today().isoformat())
    q = request.args.get('q', '').strip()
    rows = attendance_rows(selected, selected, q)
    return render_template('attendance.html', rows=rows, selected=selected, q=q)


@app.route('/reports')
@roles_required('admin', 'faculty')
def reports():
    start = request.args.get('start', date.today().replace(day=1).isoformat())
    end = request.args.get('end', date.today().isoformat())
    rows = attendance_rows(start, end, request.args.get('q','').strip())
    with connect_db() as conn:
        total = conn.execute('SELECT COUNT(*) n FROM students').fetchone()['n']
        days = conn.execute('SELECT COUNT(DISTINCT session_date) n FROM attendance WHERE session_date BETWEEN ? AND ?', (start,end)).fetchone()['n']
    return render_template('reports.html', rows=rows, start=start, end=end, total=total, days=days)


@app.route('/reports.csv')
@roles_required('admin', 'faculty')
def report_csv():
    start=request.args.get('start', date.today().isoformat()); end=request.args.get('end', start)
    output=StringIO(); writer=csv.writer(output); writer.writerow(['Student ID','Name','Date','Status','Time'])
    for r in attendance_rows(start,end,request.args.get('q','')):
        writer.writerow([r['student_id'],r['name'],r['session_date'] or start,r['status'],r['marked_at'] or ''])
    audit('REPORT_EXPORTED', f'{start} to {end}')
    resp=make_response(output.getvalue()); resp.headers['Content-Type']='text/csv; charset=utf-8'
    resp.headers['Content-Disposition']='attachment; filename=smartattend_report.csv'; return resp


@app.route('/faculty', methods=['GET','POST'])
@roles_required('admin')
def faculty():
    if request.method == 'POST':
        username=request.form.get('username','').strip().lower(); name=request.form.get('full_name','').strip(); password=request.form.get('password','')
        if not username or not name or len(password)<8:
            flash('Provide username, full name and a password of at least 8 characters.','error')
        else:
            try:
                with connect_db() as conn:
                    conn.execute('INSERT INTO users(username,full_name,password_hash,role,created_at) VALUES(?,?,?,?,?)',
                                 (username,name,generate_password_hash(password, method='pbkdf2:sha256:600000'),'faculty',datetime.now().isoformat(timespec='seconds')))
                audit('FACULTY_CREATED', username); flash('Faculty account created.','success')
            except sqlite3.IntegrityError: flash('That username is already in use.','error')
        return redirect(url_for('faculty'))
    with connect_db() as conn: users=conn.execute('SELECT id,username,full_name,role,active,created_at FROM users ORDER BY role,full_name').fetchall()
    return render_template('faculty.html', users=users)


@app.route('/faculty/<int:user_id>/toggle', methods=['POST'])
@roles_required('admin')
def toggle_faculty(user_id):
    if user_id == session.get('user_id'):
        flash('You cannot deactivate your own account.','error')
    else:
        with connect_db() as conn:
            user=conn.execute('SELECT username,active,role FROM users WHERE id=?',(user_id,)).fetchone()
            if user and user['role']=='faculty':
                conn.execute('UPDATE users SET active=? WHERE id=?',(0 if user['active'] else 1,user_id))
                audit('FACULTY_STATUS_CHANGED',user['username'])
    return redirect(url_for('faculty'))


@app.route('/audit')
@roles_required('admin')
def audit_page():
    with connect_db() as conn: logs=conn.execute('SELECT * FROM audit_logs ORDER BY id DESC LIMIT 250').fetchall()
    return render_template('audit.html', logs=logs)


@app.errorhandler(403)
def forbidden(_): return render_template('message.html', title='Access restricted', message='Your account does not have permission to open this page.'),403


if __name__ == '__main__':
    init_db()
    train_recognizer()
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT','5001')), debug=False, threaded=True)
