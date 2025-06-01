import os
from peewee import SqliteDatabase


# Application Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', '785243')
    DEBUG = True

    # Database Configuration
    DATABASE_NAME = 'Sps3.db'
    DATABASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), DATABASE_NAME)
    db = SqliteDatabase(DATABASE_PATH)

    # File Upload Configuration
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'pdf'}
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

    # Email Configuration
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = 'agueroroman27@gmail.com'
    MAIL_PASSWORD = 'your-email-password'  # Update with your password
    MAIL_DEFAULT_SENDER = 'agueroroman27@gmail.com'

    @staticmethod
    def init_app(app):
        # Ensure upload folder exists
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        # Initialize database
        from models import create_tables
        create_tables()


# Create configuration instance
config = Config()