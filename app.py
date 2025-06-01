import os
import uuid
import app
import joblib
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from flask import Flask, render_template, redirect, url_for, request, flash, session, send_file, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from email_validator import validate_email, EmailNotValidError
from flask_mail import Mail, Message
from config import config
from models import Gender, Teacher, PhoneNumber, AnalysisLog, Student, ActivityLog, Notification, DownloadLog
from peewee import DoesNotExist, IntegrityError
from collections import defaultdict
from tensorflow import keras
from io import BytesIO
from math import ceil
from datetime import datetime, timedelta, time


app = Flask(__name__)
app.secret_key = config.SECRET_KEY

app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')
mail = Mail(app)

# Initialize application
config.init_app(app)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            validate_email(email)
        except EmailNotValidError as e:
            flash(f"Invalid email format: {str(e)}", 'danger')
            return redirect(url_for('login'))

        # Check for admin credentials
        if email == 'admin@gmail.com' and password == 'admin25':
            session['teacher_id'] = 'admin'
            session['teacher_email'] = email
            session['teacher_name'] = 'Admin User'
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))

        try:
            teacher = Teacher.get(Teacher.email == email)

            if check_password_hash(teacher.password_hash, password):

                teacher.last_login = datetime.now()
                teacher.save()
                session['teacher_id'] = teacher.id
                session['teacher_email'] = teacher.email
                session['teacher_name'] = f"{teacher.first_name} {teacher.last_name}"
                flash('Login successful!', 'success')
                return redirect(url_for('user_dashboard'))
            else:
                flash('Incorrect password. Please try again.', 'danger')
        except DoesNotExist:
            flash('No account found with that email.', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        country_code = request.form['country_code']
        phone = request.form['phone']
        gender_name = request.form['gender'].lower()
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if gender_name not in ["male", "female"]:
            flash("Invalid gender selection. Please choose Male or Female.", 'danger')
            return redirect(url_for('register'))

        try:
            validate_email(email)
        except EmailNotValidError as e:
            flash(f"Invalid email format: {str(e)}", 'danger')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match. Please try again.', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        try:
            gender, created = Gender.get_or_create(name=gender_name)
            teacher = Teacher.create(
                first_name=first_name,
                last_name=last_name,
                email=email,
                gender=gender,
                password_hash=hashed_password
            )
            PhoneNumber.create(
                teacher=teacher,
                country_code=country_code,
                phone=phone
            )
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
        except IntegrityError:
            flash('An account with this email already exists. Please use a different email.', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')


@app.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    if request.method == 'POST':
        email = request.form['email']
        try:
            teacher = Teacher.get(Teacher.email == email)
            reset_token = str(uuid.uuid4())
            teacher.reset_token = reset_token
            teacher.save()

            reset_link = url_for('reset_password', token=reset_token, _external=True)
            msg = Message('Password Reset Request', sender='agueroroman27@gmail.com', recipients=[email])
            msg.body = f'Click the following link to reset your password: {reset_link}'
            mail.send(msg)

            flash('A password reset link has been sent to your email.', 'success')
            return redirect(url_for('login'))

        except DoesNotExist:
            flash('No account found with that email.', 'danger')

    return render_template('reset_password.html')


@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if request.method == 'GET':
        # Verify the token
        try:
            teacher = Teacher.get(Teacher.reset_token == token)
            return render_template('reset_password_form.html', token=token)
        except DoesNotExist:
            flash('Invalid or expired reset token.', 'danger')
            return redirect(url_for('reset_password_request'))

    elif request.method == 'POST':
        # Handle password reset form submission
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('reset_password', token=token))

        try:
            teacher = Teacher.get(Teacher.reset_token == token)
            teacher.password_hash = generate_password_hash(password)
            teacher.reset_token = None  # Clear the token after use
            teacher.save()
            flash('Your password has been updated! Please log in.', 'success')
            return redirect(url_for('login'))
        except DoesNotExist:
            flash('Invalid or expired reset token.', 'danger')
            return redirect(url_for('reset_password_request'))


@app.route('/user_dashboard')
def user_dashboard():
    if 'teacher_id' not in session:
        flash('Please log in to access the dashboard.', 'danger')
        return redirect(url_for('login'))

    try:
        teacher = Teacher.get(Teacher.id == session['teacher_id'])
        phone_numbers = PhoneNumber.select().where(PhoneNumber.teacher == teacher)
        # Get analysis statistics
        analysis_stats = {
            # Files analyzed
            'files_analyzed': AnalysisLog.select().where(AnalysisLog.teacher == teacher).count(),
            'files_analyzed_weekly': AnalysisLog.select().where(
                (AnalysisLog.teacher == teacher) &
                (AnalysisLog.created_at >= datetime.now() - timedelta(days=7))
            ).count(),
            'files_analyzed_monthly': AnalysisLog.select().where(
                (AnalysisLog.teacher == teacher) &
                (AnalysisLog.created_at >= datetime.now() - timedelta(days=30))
            ).count(),

            # Students analyzed
            'students_analyzed': Student.select().where(Student.analyzed_by == teacher).count(),
            'students_analyzed_weekly': Student.select().where(
                (Student.analyzed_by == teacher) &
                (Student.analysis_date >= datetime.now() - timedelta(days=7))
            ).count(),
            'students_analyzed_monthly': Student.select().where(
                (Student.analyzed_by == teacher) &
                (Student.analysis_date >= datetime.now() - timedelta(days=30))
            ).count(),

            # At-risk students
            'at_risk_students': Student.select().where(
                (Student.analyzed_by == teacher) &
                (Student.at_risk == True)
            ).count(),
            'new_at_risk_weekly': Student.select().where(
                (Student.analyzed_by == teacher) &
                (Student.at_risk == True) &
                (Student.analysis_date >= datetime.now() - timedelta(days=7))
            ).count(),
            'new_at_risk_monthly': Student.select().where(
                (Student.analyzed_by == teacher) &
                (Student.at_risk == True) &
                (Student.analysis_date >= datetime.now() - timedelta(days=30))
            ).count(),

            # Downloads
            'downloads_total': DownloadLog.select().where(DownloadLog.teacher == teacher).count(),
            'downloads_weekly': DownloadLog.select().where(
                (DownloadLog.teacher == teacher) &
                (DownloadLog.created_at >= datetime.now() - timedelta(days=7))
            ).count(),
            'downloads_monthly': DownloadLog.select().where(
                (DownloadLog.teacher == teacher) &
                (DownloadLog.created_at >= datetime.now() - timedelta(days=30))
            ).count()
        }

        # Get recent at-risk students
        recent_at_risk = Student.select().where(
            (Student.analyzed_by == teacher) &
            (Student.at_risk == True)
        ).order_by(Student.analysis_date.desc()).limit(5)

        # Get performance distribution data - actual query
        performance_bins = [0, 20, 40, 60, 80, 100]
        performance_distribution = []
        for i in range(len(performance_bins) - 1):
            lower = performance_bins[i]
            upper = performance_bins[i + 1]
            count = Student.select().where(
                (Student.analyzed_by == teacher) &
                (Student.predicted_mean >= lower) &
                (Student.predicted_mean < upper)
            ).count()
            performance_distribution.append(count)

        # Get weak subjects data - actual query
        weak_subjects_data = []
        weak_subjects_labels = app.config['COMPULSORY_SUBJECTS']  # Focus on core subjects

        # Count how many students have each subject as weak (< threshold)
        for subject in weak_subjects_labels:
            count = Student.select().where(
                (Student.analyzed_by == teacher) &
                (Student.weak_subjects.contains(f'"{subject}"'))  # Replaces JSON_CONTAINS
            ).count()
            weak_subjects_data.append(count)

        # Get recent activities
        recent_activities = ActivityLog.select().where(
            ActivityLog.teacher == teacher
        ).order_by(ActivityLog.timestamp.desc()).limit(5)

        # Get unread notifications count
        unread_notifications = Notification.select().where(
            (Notification.teacher == teacher) &
            (Notification.is_read == False)
        ).count()

        return render_template(
            'user_dashboard.html',
            teacher_name=f"{teacher.first_name} {teacher.last_name}",
            teacher_email=teacher.email,
            phone_numbers=phone_numbers,
            last_login=teacher.last_login.strftime('%Y-%m-%d %H:%M') if teacher.last_login else 'Never',
            analysis_stats=analysis_stats,
            recent_at_risk=recent_at_risk,
            performance_distribution=performance_distribution,
            weak_subjects_labels=weak_subjects_labels,
            weak_subjects_data=weak_subjects_data,
            recent_activities=recent_activities,
            unread_notifications=unread_notifications
        )
    except DoesNotExist:
        flash('User information not found.', 'danger')
        return redirect(url_for('login'))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'teacher_id' not in session:
        flash('Please log in to access your profile.', 'danger')
        return redirect(url_for('login'))

    try:
        teacher = Teacher.get(Teacher.id == session['teacher_id'])
        phone_number = PhoneNumber.get_or_none(PhoneNumber.teacher == teacher)
        teacher_name = f"{teacher.first_name} {teacher.last_name}"

        if request.method == 'POST':
            teacher.first_name = request.form['first_name']
            teacher.last_name = request.form['last_name']
            teacher.email = request.form['email']
            teacher.gender = Gender.get(Gender.name == request.form['gender'].lower())

            if phone_number:
                phone_number.country_code = request.form['country_code']
                phone_number.phone = request.form['phone']
                phone_number.save()
            else:
                PhoneNumber.create(
                    teacher=teacher,
                    country_code=request.form['country_code'],
                    phone=request.form['phone']
                )

            teacher.save()
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))

        return render_template('profile.html',
                            user=teacher,
                            phone=phone_number,
                            teacher_name=teacher_name)
    except DoesNotExist:
        flash('User information not found.', 'danger')
        return redirect(url_for('login'))


@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'teacher_id' not in session:
        flash('Please log in to edit your profile.', 'danger')
        return redirect(url_for('login'))

    try:
        teacher = Teacher.get(Teacher.id == session['teacher_id'])
        phone_number = PhoneNumber.get_or_none(PhoneNumber.teacher == teacher)

        if request.method == 'POST':
            teacher.first_name = request.form['first_name']
            teacher.last_name = request.form['last_name']
            teacher.email = request.form['email']
            teacher.gender = Gender.get(Gender.name == request.form['gender'].lower())
            teacher.save()

            if phone_number:
                phone_number.country_code = request.form['country_code']
                phone_number.phone = request.form['phone']
                phone_number.save()
            else:
                PhoneNumber.create(
                    teacher=teacher,
                    country_code=request.form['country_code'],
                    phone=request.form['phone']
                )

            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))

        return render_template('edit_profile.html', user=teacher, phone=phone_number)
    except DoesNotExist:
        flash('User information not found.', 'danger')
        return redirect(url_for('login'))


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'teacher_id' not in session:
        flash('Please log in to access settings.', 'danger')
        return redirect(url_for('login'))

    teacher = Teacher.get(Teacher.id == session['teacher_id'])

    if request.method == 'POST':
        # Handle form submissions (e.g., password change, email update)
        if 'change_password' in request.form:
            current_password = request.form['current_password']
            new_password = request.form['new_password']

            if check_password_hash(teacher.password_hash, current_password):
                teacher.password_hash = generate_password_hash(new_password)
                teacher.save()
                flash('Password updated successfully!', 'success')
            else:
                flash('Current password is incorrect.', 'danger')

        elif 'update_email' in request.form:
            new_email = request.form['new_email']
            # Add email validation and update logic here

    return render_template('settings.html', user=teacher)


@app.route('/admin/dashboard')
def admin_dashboard():
    if 'teacher_id' in session:

        if session['teacher_email'] == 'admin@gmail.com':
            # Get user count
            user_count = Teacher.select().count()

            return render_template(
                'admin_dashboard.html',
                teacher_name='Admin User',
                user_count=user_count
            )
        else:
            teacher = Teacher.get(Teacher.id == session['teacher_id'])
            return render_template('admin_dashboard.html', teacher_name=teacher.fullname)
    else:
        flash('Please log in to access the admin dashboard.', 'danger')
        return redirect(url_for('login'))


@app.route('/admin/users')
def admin_users():
    if 'teacher_id' in session:
        # Check if the user is the admin
        if session['teacher_email'] == 'admin@gmail.com':
            search_query = request.args.get('search', '')
            sort_field = request.args.get('sort', 'id')
            sort_order = request.args.get('order', 'asc')

            # Start with base query
            query = Teacher.select()

            # Apply search filter if search query exists
            if search_query:
                query = query.where(
                    (Teacher.first_name.contains(search_query)) |
                    (Teacher.last_name.contains(search_query)) |
                    (Teacher.email.contains(search_query))
                )

            # Apply sorting
            if sort_field in ['first_name', 'last_name', 'email']:
                if sort_order == 'asc':
                    query = query.order_by(getattr(Teacher, sort_field).asc())
                else:
                    query = query.order_by(getattr(Teacher, sort_field).desc())

            users = query

            return render_template(
                'admin_users.html',
                users=users,
                search_query=search_query
            )
        else:
            teacher = Teacher.get(Teacher.id == session['teacher_id'])
            return render_template('admin_users.html', users=[teacher])
    else:
        flash('Please log in to access the admin dashboard.', 'danger')
        return redirect(url_for('login'))


@app.route('/admin/users/edit/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    if 'teacher_id' in session:
        if session['teacher_email'] == 'admin@gmail.com':
            try:
                user = Teacher.get(Teacher.id == user_id)
                phone_number = PhoneNumber.get(PhoneNumber.teacher == user)

                if request.method == 'POST':
                    user.first_name = request.form['first_name']
                    user.last_name = request.form['last_name']
                    user.email = request.form['email']
                    user.gender = Gender.get(Gender.name == request.form['gender'].lower())
                    phone_number.country_code = request.form['country_code']
                    phone_number.phone = request.form['phone']

                    user.save()
                    phone_number.save()
                    flash('User details updated successfully!', 'success')
                    return redirect(url_for('admin_users'))

                return render_template('edit_user.html', user=user, phone=phone_number)
            except DoesNotExist:
                flash('User not found.', 'danger')
                return redirect(url_for('admin_users'))
        else:
            teacher = Teacher.get(Teacher.id == session['teacher_id'])
            return render_template('admin_users.html', users=[teacher])
    else:
        flash('Please log in to access the admin dashboard.', 'danger')
        return redirect(url_for('login'))


@app.route('/admin/users/delete/<int:user_id>', methods=['GET', 'POST'])
def delete_user(user_id):
    if 'teacher_id' not in session:
        flash('Please log in to access the admin dashboard.', 'danger')
        return redirect(url_for('login'))

    # Verify admin privileges
    if session.get('teacher_email') != 'admin@gmail.com':
        flash('Access denied. You are not authorized.', 'danger')
        return redirect(url_for('user_dashboard'))

    try:
        user = Teacher.get(Teacher.id == user_id)

        if request.method == 'POST':

            PhoneNumber.delete().where(PhoneNumber.teacher == user).execute()

            user.delete_instance()
            flash('User deleted successfully!', 'success')
            return redirect(url_for('admin_users'))

        return render_template('delete_user.html', user=user)

    except DoesNotExist:
        flash('User not found.', 'danger')
        return redirect(url_for('admin_users'))


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')

    response = redirect(url_for('login'))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


@app.after_request
def add_cache_control(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
class Config:
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv'}
    MODEL_DIR = 'TrainedModels'
    COMPULSORY_SUBJECTS = ['Eng', 'Kiswahili', 'Maths', 'Chemistry']
    OPTIONAL_SUBJECTS = ['Biology', 'Physics', 'History', 'Geo', 'CRE', 'IRE',
                         'Agriculture', 'Computer', 'French', 'Business']
    ALL_SUBJECTS = COMPULSORY_SUBJECTS + OPTIONAL_SUBJECTS
    GRADE_C_THRESHOLD = 50
    MIN_SUBJECTS = 7
    RESULTS_PER_PAGE = 80
    TOTAL_EXPECTED_FEATURES = 61  # From the model implementation


app.config.from_object(Config)


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_DIR'], exist_ok=True)
os.makedirs(os.path.join('static', 'temp_plots'), exist_ok=True)

# Load model and scaler
try:
    scaler = joblib.load(os.path.join(app.config['MODEL_DIR'], 'scaler.joblib'))
    subjects_list = joblib.load(os.path.join(app.config['MODEL_DIR'], 'subjects.joblib'))

    model_path_h5 = os.path.join(app.config['MODEL_DIR'], 'best_model.h5')
    if os.path.exists(model_path_h5):
        model = keras.models.load_model(model_path_h5)
        logger.info("Loaded neural network model")
    else:
        # Fall back to traditional model
        model_path_joblib = os.path.join(app.config['MODEL_DIR'], 'best_model.joblib')
        model = joblib.load(model_path_joblib)
        logger.info("Loaded traditional ML model")

except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Could not load prediction model")


# File Upload Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}
app.config['MAX_FILE_SIZE'] = 16 * 1024 * 1024  # 16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def secure_filename_with_timestamp(filename):
    """Add timestamp to filename to prevent collisions"""
    from datetime import datetime
    name, ext = os.path.splitext(secure_filename(filename))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{timestamp}{ext}"


def cleanup_old_uploads():
    """Delete files older than 24 hours from uploads folder"""
    now = datetime.now().timestamp()
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(filepath) and os.path.getmtime(filepath) < now - 24*3600:
            os.remove(filepath)


def validate_input_data(form_data):
    """Validate all form inputs for manual analysis"""
    errors = []

    # Validate behavior score
    try:
        behavior = float(form_data.get('Behavior', 3))
        if not (1 <= behavior <= 5):
            errors.append("Behavior score must be between 1 and 5")
    except ValueError:
        errors.append("Invalid behavior score")

    # Validate attendance
    try:
        attendance = float(form_data.get('Attendance', 80))
        if not (0 <= attendance <= 100):
            errors.append("Attendance must be between 0 and 100%")
    except ValueError:
        errors.append("Invalid attendance percentage")

    # Validate subject scores
    for subject in app.config['ALL_SUBJECTS']:
        try:
            score = float(form_data.get(subject, 50))
            if not (0 <= score <= 100):
                errors.append(f"{subject} score must be between 0 and 100")
        except ValueError:
            errors.append(f"Invalid score for {subject}")

    return errors


def clean_data(df, exam_type):
    """Clean and preprocess the dataset using the updated implementation"""
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df.columns = df.columns.str.strip().str.replace(r'\(.*\)', '', regex=True).str.replace('%', '').str.strip()
    df['Exam_Type'] = exam_type

    # Handle attendance
    if 'Attendance' in df.columns:
        if isinstance(df['Attendance'].iloc[0], str):
            df['Attendance'] = df['Attendance'].str.replace('%', '').astype(float)
        df['Attendance'] = np.clip(df['Attendance'], 0, 100)
        attendance_median = df['Attendance'].median()
        df['Attendance'] = df['Attendance'].replace(0, attendance_median)
    else:
        df['Attendance'] = 100

    # Handle behavior
    if 'Behavior' in df.columns:
        df['Behavior'] = np.clip(df['Behavior'], 1, 5)
    else:
        df['Behavior'] = 3

    # Process subject marks
    for subject in app.config['ALL_SUBJECTS']:
        if subject in df.columns:
            df[subject] = pd.to_numeric(df[subject], errors='coerce').fillna(0)

    # Calculate metrics
    subject_cols = [s for s in app.config['ALL_SUBJECTS'] if s in df.columns]
    df['Subjects_Taken'] = df[subject_cols].notna().sum(axis=1)
    df = df[df['Subjects_Taken'] >= app.config['MIN_SUBJECTS']]
    df['Mean_Marks'] = df[subject_cols].mean(axis=1)

    # Identify weak subjects
    for subject in subject_cols:
        df[f'{subject}_Weak'] = (df[subject] < app.config['GRADE_C_THRESHOLD']).astype(int)

    df['Weak_Subjects_Count'] = df[[f'{s}_Weak' for s in subject_cols]].sum(axis=1)
    df['At_Risk'] = (df['Mean_Marks'] < app.config['GRADE_C_THRESHOLD']).astype(int)

    return df


def create_features(df, is_training=False):
    """Create features with consistent dimensions using the updated implementation"""
    subject_cols = [s for s in app.config['ALL_SUBJECTS'] if s in df.columns]
    features = []
    student_info = []

    # Prediction mode (no historical data)
    for _, row in df.iterrows():
        # Base features (14 subjects + 5 basic)
        feature_vec = row[subject_cols].tolist() + [
            row['Mean_Marks'],
            row['Weak_Subjects_Count'],
            row['Behavior'],
            row['Attendance'],
            0  # No previous exams
        ]

        # Pad with zeros for trend features (42 zeros)
        feature_vec.extend([0] * (len(subject_cols) * 3))

        # Ensure we have exactly 61 features
        if len(feature_vec) < app.config['TOTAL_EXPECTED_FEATURES']:
            feature_vec.extend([0] * (app.config['TOTAL_EXPECTED_FEATURES'] - len(feature_vec)))

        features.append(feature_vec)
        student_info.append({
            'Adm_No': row['Adm No'],
            'Name': row['Name'],
            'Exam_Type': row['Exam_Type'],
            'Current_Mean': row['Mean_Marks'],
            'Weak_Subjects': [s for s in subject_cols if row[s] < app.config['GRADE_C_THRESHOLD']],
            'Behavior': row['Behavior'],
            'Attendance': row['Attendance'],
            'Previous_Exams_Count': 0
        })

    features_array = np.array(features)
    features_array = np.nan_to_num(features_array)

    if len(features_array) == 0:
        raise ValueError("No valid samples created - check input data")

    return features_array, student_info


def evaluate_student(features, student_info):
    """Evaluate a student's performance using the updated implementation"""
    if len(features) < app.config['TOTAL_EXPECTED_FEATURES']:
        padded_data = np.zeros(app.config['TOTAL_EXPECTED_FEATURES'])
        padded_data[:len(features)] = features
        features = padded_data

    features_scaled = scaler.transform([features])

    if isinstance(model, keras.Model):
        prediction = model.predict(features_scaled).flatten()[0]
    else:
        prediction = model.predict(features_scaled)[0]

        # Debugging output
    logger.info(f"Student: {student_info['Name']}")
    logger.info(f"Raw prediction: {prediction}")
    logger.info(f"Threshold: {app.config['GRADE_C_THRESHOLD']}")

    # Calculate at_risk status
    at_risk = prediction < app.config['GRADE_C_THRESHOLD']
    logger.info(f"At risk: {at_risk}")

    # Identify weak subjects
    weak_subjects = student_info['Weak_Subjects']

    # Determine priority subjects
    subject_gaps = {}
    for i, subj in enumerate([s for s in app.config['ALL_SUBJECTS'] if s in student_info]):
        current_score = features[i]
        gap = app.config['GRADE_C_THRESHOLD'] - current_score
        if gap > 0:
            subject_gaps[subj] = gap

    priority_subjects = sorted(subject_gaps.items(), key=lambda x: x[1], reverse=True)[:3]

    # Format behavior and attendance
    behavior_score = max(1, min(5, round(student_info['Behavior'], 1)))
    attendance_score = max(50, min(100, round(student_info['Attendance'], 1)))

    return {
        'Adm_No': student_info['Adm_No'],
        'Name': student_info['Name'],
        'Exam_Type': student_info['Exam_Type'],
        'Current_Mean': round(student_info['Current_Mean'], 1),
        'Predicted_Mean': round(prediction, 1),
        'At_Risk': 'YES' if at_risk else 'NO',
        'Weak_Subjects': weak_subjects,
        'Priority_Subjects': [subj[0] for subj in priority_subjects],
        'Behavior': behavior_score,
        'Behavior_Display': f"{behavior_score:.1f}/5",
        'Attendance': attendance_score,
        'Attendance_Display': f"{attendance_score:.1f}%"
    }


def generate_visualizations(results):
    """Generate performance visualizations"""
    plot_urls = {}
    plot_dir = os.path.join('static', 'temp_plots')

    try:
        # Performance Distribution
        plt.figure(figsize=(10, 6))
        scores = [res['Predicted_Mean'] for res in results]
        plt.hist(scores, bins=10, color='blue', alpha=0.7, edgecolor='black')
        plt.axvline(x=app.config['GRADE_C_THRESHOLD'], color='red', linestyle='--', label='At-Risk Threshold')
        plt.title('Predicted Performance Distribution')
        plt.xlabel('Mean Score')
        plt.ylabel('Number of Students')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(plot_dir, 'performance_dist.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_urls['performance_dist'] = plot_path.replace('\\', '/')

        # Weak Subjects Analysis
        weak_counts = defaultdict(int)
        for res in results:
            for subject in res['Weak_Subjects']:
                weak_counts[subject] += 1

        if weak_counts:
            plt.figure(figsize=(12, 6))
            subjects = list(weak_counts.keys())
            counts = list(weak_counts.values())
            plt.bar(subjects, counts, color='red', alpha=0.7, edgecolor='black')
            plt.title('Weak Subjects Distribution Among At-Risk Students')
            plt.xlabel('Subjects')
            plt.ylabel('Number of Students')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plot_path = os.path.join(plot_dir, 'weak_subjects.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_urls['weak_subjects'] = plot_path.replace('\\', '/')

    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")

    return plot_urls


def create_download_report(results, format='excel'):
    """Create downloadable report in specified format"""
    try:
        if not results:
            raise ValueError("No results to export")

        # Create DataFrame from results
        report_data = []
        for res in results:
            report_data.append({
                'Adm No': res.get('Adm_No', ''),
                'Name': res.get('Name', ''),
                'Exam Type': res.get('Exam_Type', ''),
                'Current Mean': res.get('Current_Mean', 0),
                'Predicted Mean': res.get('Predicted_Mean', 0),
                'At Risk': res.get('At_Risk', 'NO'),
                'Weak Subjects': ', '.join(res.get('Weak_Subjects', [])),
                'Priority Subjects': ', '.join(res.get('Priority_Subjects', [])),
                'Behavior': res.get('Behavior_Display', ''),
                'Attendance': res.get('Attendance_Display', '')
            })

        df = pd.DataFrame(report_data)

        if format == 'csv':
            output = BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return output, 'text/csv', 'student_performance_report.csv'
        else:
            # Default to Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Student Analysis', index=False)

                # Formatting
                workbook = writer.book
                worksheet = writer.sheets['Student Analysis']

                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#4472C4',
                    'font_color': 'white',
                    'border': 1
                })

                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                # Add conditional formatting for at-risk students
                at_risk_format = workbook.add_format({'bg_color': '#FFC7CE'})
                worksheet.conditional_format(1, 5, len(df), 5, {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': 'Yes',
                    'format': at_risk_format
                })

                for i, column in enumerate(df.columns):
                    column_width = max(df[column].astype(str).map(len).max(), len(column)) + 2
                    worksheet.set_column(i, i, min(column_width, 50))

            output.seek(0)
            return output, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'student_performance_report.xlsx'

    except Exception as e:
        logger.error(f"Error creating report: {str(e)}")
        raise


@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if 'teacher_id' not in session:
        flash('Please log in to access analysis tools.', 'danger')
        return redirect(url_for('login'))

    teacher = Teacher.get(Teacher.id == session['teacher_id'])
    results = []
    plot_urls = {}
    analyzed = False
    page = request.args.get('page', 1, type=int)

    if request.method == 'POST':
        # Handle file upload for batch analysis
        if 'analysis_file' in request.files:
            file = request.files['analysis_file']
            exam_type = request.form.get('exam_type', 'Term 1 Opener')

            if file.filename == '':
                flash('No file selected', 'danger')
                return render_template('analysis.html', analyzed=False)

            if not allowed_file(file.filename):
                flash('Invalid file type. Only CSV or Excel files are allowed.', 'danger')
                return redirect(url_for('analysis'))

            if file.content_length > app.config['MAX_FILE_SIZE']:
                flash('File too large (max 16MB)', 'danger')
                return redirect(url_for('analysis'))

            try:
                # Create activity log
                activity = ActivityLog.create(
                    teacher=teacher,
                    activity_type='analysis',
                    details=f"Started batch analysis of {file.filename}",
                    status='pending'
                )

                # Secure filename with timestamp and save
                filename = secure_filename_with_timestamp(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Clean up old uploads
                cleanup_old_uploads()

                # Load and process the file
                try:
                    if filename.endswith('.csv'):
                        df = pd.read_csv(filepath)
                    else:  # Excel
                        df = pd.read_excel(filepath)
                except Exception as e:
                    flash(f'Error reading file: {str(e)}', 'danger')
                    os.remove(filepath)
                    return redirect(url_for('analysis'))

                df = clean_data(df, exam_type)
                features, student_info_list = create_features(df)

                # Process each student
                for idx, student_info in enumerate(student_info_list):
                    student_features = features[idx]
                    result = evaluate_student(student_features, student_info)

                    # Check if student already exists
                    try:
                        existing_student = Student.get(Student.adm_no == result['Adm_No'])
                        # Update existing record
                        existing_student.name = result['Name']
                        existing_student.exam_type = result['Exam_Type']
                        existing_student.current_mean = result['Current_Mean']
                        existing_student.predicted_mean = result['Predicted_Mean']
                        existing_student.at_risk = result['At_Risk'] == 'YES'
                        existing_student.weak_subjects = ", ".join(result['Weak_Subjects'])
                        existing_student.priority_subjects = ", ".join(result['Priority_Subjects'])
                        existing_student.behavior_score = result['Behavior']
                        existing_student.attendance_percentage = result['Attendance']
                        existing_student.analyzed_by = teacher
                        existing_student.save()
                    except DoesNotExist:
                        # Create new record if doesn't exist
                        Student.create(
                            adm_no=result['Adm_No'],
                            name=result['Name'],
                            exam_type=result['Exam_Type'],
                            current_mean=result['Current_Mean'],
                            predicted_mean=result['Predicted_Mean'],
                            at_risk=result['At_Risk'] == 'YES',
                            weak_subjects=", ".join(result['Weak_Subjects']),
                            priority_subjects=", ".join(result['Priority_Subjects']),
                            behavior_score=result['Behavior'],
                            attendance_percentage=result['Attendance'],
                            analyzed_by=teacher
                        )

                    results.append(result)

                # Generate visualizations
                plot_urls = generate_visualizations(results)
                analyzed = True

                # Create analysis log
                AnalysisLog.create(
                    teacher=teacher,
                    filename=filename,
                    filepath=filepath,
                    exam_type=exam_type,
                    students_processed=len(results),
                    at_risk_count=sum(1 for r in results if r['At_Risk'] == 'YES')
                )

                # Update activity log
                activity.details = f"Completed analysis of {filename} with {len(results)} students"
                activity.status = 'completed'
                activity.save()

                flash('Batch analysis completed successfully!', 'success')

            except Exception as e:
                logger.error(f"Analysis error: {str(e)}")
                if 'filepath' in locals() and os.path.exists(filepath):
                    os.remove(filepath)
                if 'activity' in locals():
                    activity.details = f"Failed analysis: {str(e)}"
                    activity.status = 'failed'
                    activity.save()
                flash(f'Analysis failed: {str(e)}', 'danger')

        # Handle manual student analysis
        elif 'manual_analysis' in request.form:
            # Validate inputs first
            validation_errors = validate_input_data(request.form)
            if validation_errors:
                for error in validation_errors:
                    flash(error, 'danger')
            else:
                try:
                    # Create activity log
                    activity = ActivityLog.create(
                        teacher=teacher,
                        activity_type='manual_analysis',
                        details="Started manual student analysis",
                        status='pending'
                    )

                    # Create student data dictionary
                    student_data = {
                        'Adm No': request.form.get('Adm_No', 'MANUAL_' + str(len(results) + 1)),
                        'Name': request.form.get('Name', 'Manual Student'),
                        'Exam_Type': request.form.get('exam_type', 'Term 1 Opener'),
                        'Behavior': float(request.form.get('Behavior', 3)),
                        'Attendance': float(request.form.get('Attendance', 80))
                    }

                    # Add all subject scores and calculate mean
                    subject_scores = []
                    for subject in app.config['ALL_SUBJECTS']:
                        score = float(request.form.get(subject, 50))
                        student_data[subject] = score
                        subject_scores.append(score)

                    # Calculate derived metrics
                    student_data['Mean_Marks'] = np.mean(subject_scores)
                    student_data['Weak_Subjects_Count'] = sum(1 for score in subject_scores
                                                              if score < app.config['GRADE_C_THRESHOLD'])

                    # Create a single-row DataFrame for feature creation
                    df = pd.DataFrame([student_data])

                    # Create features using the same pipeline
                    features, student_info_list = create_features(df)

                    if len(features) > 0 and len(student_info_list) > 0:
                        result = evaluate_student(features[0], student_info_list[0])
                        results.append(result)

                        # Save student to database
                        Student.create(
                            adm_no=result['Adm_No'],
                            name=result['Name'],
                            exam_type=result['Exam_Type'],
                            current_mean=result['Current_Mean'],
                            predicted_mean=result['Predicted_Mean'],
                            at_risk=result['At_Risk'] == 'YES',
                            weak_subjects=result['Weak_Subjects'],
                            priority_subjects=result['Priority_Subjects'],
                            behavior_score=result['Behavior'],
                            attendance_percentage=result['Attendance'],
                            analyzed_by=teacher
                        )

                        # Generate visualizations (even for single student)
                        plot_urls = generate_visualizations(results)
                        analyzed = True

                        # Update activity log
                        activity.details = f"Completed manual analysis for {result['Name']}"
                        activity.status = 'completed'
                        activity.save()

                        flash('Manual student analysis completed!', 'success')
                    else:
                        activity.details = "Error creating features for manual student"
                        activity.status = 'failed'
                        activity.save()
                        flash('Error creating features for manual student', 'danger')

                except Exception as e:
                    logger.error(f"Error in manual analysis: {str(e)}")
                    if 'activity' in locals():
                        activity.details = f"Error in manual analysis: {str(e)}"
                        activity.status = 'failed'
                        activity.save()
                    flash(f'Error analyzing student: {str(e)}', 'danger')

    # Calculate summary statistics
    avg_predicted = 0
    at_risk_count = 0
    if results:
        avg_predicted = round(sum(r['Predicted_Mean'] for r in results) / len(results), 1)
        at_risk_count = sum(1 for r in results if r['At_Risk'] == 'YES')

    # Pagination - Now query from database instead of results list
    query = Student.select().where(Student.analyzed_by == teacher).order_by(Student.analysis_date.desc())
    total_students = query.count()
    total_pages = ceil(total_students / app.config['RESULTS_PER_PAGE'])

    # Get paginated results from database
    paginated_students = query.paginate(page, app.config['RESULTS_PER_PAGE'])

    # Convert to format expected by template
    paginated_results = [{
        'Adm_No': student.adm_no,
        'Name': student.name,
        'Exam_Type': student.exam_type,
        'Current_Mean': student.current_mean,
        'Predicted_Mean': student.predicted_mean,
        'At_Risk': 'YES' if student.at_risk else 'NO',
        'Weak_Subjects': student.weak_subjects.split(', ') if student.weak_subjects else [],
        'Priority_Subjects': student.priority_subjects.split(', ') if student.priority_subjects else [],
        'Behavior': student.behavior_score,
        'Behavior_Display': f"{student.behavior_score:.1f}/5",
        'Attendance': student.attendance_percentage,
        'Attendance_Display': f"{student.attendance_percentage:.1f}%"
    } for student in paginated_students]

    return render_template(
        'analysis.html',
        analyzed=analyzed,
        results=paginated_results,
        plot_urls=plot_urls,
        avg_predicted=avg_predicted,
        at_risk_count=at_risk_count,
        subjects=app.config['ALL_SUBJECTS'],
        compulsory_subjects=app.config['COMPULSORY_SUBJECTS'],
        current_page=page,
        total_pages=total_pages
    )


@app.route('/download_report', methods=['POST'])
def download_report():
    if 'teacher_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    teacher = Teacher.get(Teacher.id == session['teacher_id'])
    results = request.json.get('results', [])

    if not results:
        return jsonify({'error': 'No results to download'}), 400

    format = request.json.get('format', 'excel')
    file, mimetype, filename = create_download_report(results, format)

    # Log the download
    DownloadLog.create(
        teacher=teacher,
        filename=filename,
        download_type='report'
    )

    return send_file(
        file,
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename
    )


if __name__ == '__main__':
    app.run(debug=True)