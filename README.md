# SmartAttendAI — Redesigned Flask Application

Mixed dashboard UI: dark navigation rail + light workspace. Roles: Admin and Faculty. Includes student face enrollment, a public check-in kiosk, protected attendance records and reports, admin-managed faculty accounts, and admin-only audit history.

## Important: preserve your existing data

The source upload for this redesign did not include your database or face images. Before using this build with your current project, copy your existing `data/smartattend.db` and the complete `data/faces/` directory into this project's `data/` directory. Keep a separate backup first. Do not replace these with the empty `data/` directory in this package.

The application retains the original `students` and `attendance` table definitions and only creates the additional `users` and `audit_logs` tables (plus indexes). It does not drop or recreate student/attendance tables. Existing face image folders are not cleared during enrollment. If you use a different database filename, set `SMARTATTEND_DB` to its path.

## macOS / VS Code setup

1. Open this folder in VS Code.
2. Copy your existing database and `faces` folder into `data/` (after backing them up).
3. In the VS Code terminal run:

   ```bash
   chmod +x run_mac.sh
   ./run_mac.sh
   ```

4. Open `http://127.0.0.1:5001/setup` and create the first administrator account. This setup page is disabled after the first account exists.
5. Sign in. The admin can create faculty accounts from **Faculty accounts**.
6. Open **Attendance kiosk** on the device used for check-in. The kiosk has no student/attendance-record browsing interface.

If macOS asks for camera access, allow it for the terminal/VS Code process and reload the kiosk. Port 5001 is used by default; set `PORT=5002` if required.

## Access model

- Admin + Faculty: dashboard, student directory, face enrollment, kiosk, attendance records, reports, CSV export.
- Admin only: faculty account creation/activation and audit trail.
- Kiosk: live face-recognition stream only; no student list or attendance report endpoints are exposed in the kiosk UI.
- Students: no login or write access to attendance records.

Passwords are stored using Werkzeug password hashing. Authenticated POST requests use a session CSRF token. Session cookies are HttpOnly and SameSite=Lax. Set a long random `SMARTATTEND_SECRET` environment variable for a persistent deployment; the default generated secret changes on restart, which signs users out.

## Important prototype limitations

This is a local academic prototype, not a production identity system. LBPH is a basic face-recognition approach and can make false matches; provide faculty/manual verification. The current kiosk marks attendance on a recognized match with LBPH distance under 65, as in the uploaded source. Add liveness detection, rate limiting, HTTPS, password reset/change, stronger session controls, backups, and a formal privacy/consent process before any real institutional deployment. Emotion/expression analysis was removed from the attendance flow because it is not needed to establish attendance.

## Existing database note

The original code stores the SQLite DB at `data/smartattend.db`. The app initializes the same student and attendance tables using `CREATE TABLE IF NOT EXISTS`; it adds the account/audit tables without deleting existing records. Always keep a dated backup before testing schema changes.


### Branding update
The interface now uses the SmartAttendAI face-recognition logo. The compact face-scan emblem is used in the sidebar and kiosk; the full generated wordmark banner appears on the administrator setup and sign-in screens. Both image assets are stored in `static/`.


## Browser-based face enrollment update

Face enrollment now uses the browser's `getUserMedia` camera stream on the enrollment page. Click **Enroll Student** to capture five frames from the visible preview and upload them to Flask. The server saves centered grayscale face crops under `data/faces/<student_id>/`; it does not open a second OpenCV camera capture session for enrollment. Camera access is available on localhost in supported browsers. Allow camera permission for the browser when prompted.

The sample capture step is intentionally independent of Haar face detection so a detector miss does not block enrollment. Position one face within the on-screen oval and keep it centered. The saved images are then used by the existing LBPH training pipeline.
