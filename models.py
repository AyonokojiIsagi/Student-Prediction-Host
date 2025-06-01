from peewee import *
from config import config
from peewee import TextField
from datetime import datetime, time

db = config.db


class BaseModel(Model):
    class Meta:
        database = db


class Gender(BaseModel):
    id = AutoField()
    name = CharField(unique=True, choices=[('male', 'Male'), ('female', 'Female')])
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    def save(self, *args, **kwargs):
        self.updated_at = datetime.now()
        return super().save(*args, **kwargs)


class Teacher(BaseModel):
    id = AutoField()
    first_name = CharField(max_length=50)
    last_name = CharField(max_length=50)
    email = CharField(unique=True, max_length=255)
    gender = ForeignKeyField(Gender, backref='teachers')
    password_hash = CharField(max_length=255)
    reset_token = CharField(null=True)
    is_admin = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)
    last_login = DateTimeField(null=True)

    @property
    def fullname(self):
        return f"{self.first_name} {self.last_name}"

    def save(self, *args, **kwargs):
        self.updated_at = datetime.now()
        return super().save(*args, **kwargs)


class PhoneNumber(BaseModel):
    id = AutoField()
    teacher = ForeignKeyField(Teacher, backref='phone_numbers')
    country_code = CharField(max_length=5)
    phone = CharField(max_length=20)
    is_primary = BooleanField(default=True)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    def save(self, *args, **kwargs):
        self.updated_at = datetime.now()
        return super().save(*args, **kwargs)


class Student(BaseModel):
    id = AutoField()
    adm_no = CharField(max_length=50, unique=True)  # Admission number
    name = CharField(max_length=100)
    exam_type = CharField(max_length=50)  # e.g., 'Term 1 Opener'
    current_mean = FloatField()
    predicted_mean = FloatField()
    at_risk = BooleanField()
    weak_subjects = TextField()
    priority_subjects = TextField()
    behavior_score = FloatField()  # 1-5 scale
    attendance_percentage = FloatField()  # 0-100%
    analyzed_by = ForeignKeyField(Teacher, backref='analyzed_students')
    analysis_date = DateTimeField(default=datetime.now)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    def save(self, *args, **kwargs):
        self.updated_at = datetime.now()
        return super().save(*args, **kwargs)


class AnalysisLog(BaseModel):
    id = AutoField()
    teacher = ForeignKeyField(Teacher, backref='analysis_logs')
    filename = CharField(max_length=255)
    filepath = CharField(max_length=512)
    exam_type = CharField(max_length=50)
    students_processed = IntegerField()
    at_risk_count = IntegerField()
    analysis_date = DateTimeField(default=datetime.now)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    def save(self, *args, **kwargs):
        self.updated_at = datetime.now()
        return super().save(*args, **kwargs)


class ActivityLog(BaseModel):
    id = AutoField()
    teacher = ForeignKeyField(Teacher, backref='activity_logs')
    activity_type = CharField(max_length=50)  # e.g., 'analysis', 'download'
    details = TextField()
    status = CharField(max_length=20, choices=[
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('pending', 'Pending')
    ])
    timestamp = DateTimeField(default=datetime.now)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    def save(self, *args, **kwargs):
        self.updated_at = datetime.now()
        return super().save(*args, **kwargs)


class Notification(BaseModel):
    id = AutoField()
    teacher = ForeignKeyField(Teacher, backref='notifications')
    message = TextField()
    is_read = BooleanField(default=False)
    notification_type = CharField(max_length=50, choices=[
        ('system', 'System'),
        ('analysis', 'Analysis'),
        ('alert', 'Alert')
    ])
    created_at = DateTimeField(default=datetime.now)
    read_at = DateTimeField(null=True)

    def mark_as_read(self):
        self.is_read = True
        self.read_at = datetime.now()
        self.save()


class DownloadLog(BaseModel):
    teacher = ForeignKeyField(Teacher, backref='downloads')
    filename = CharField()
    download_type = CharField()  # 'report' or 'data'
    created_at = DateTimeField(default=datetime.now)

    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)


def create_tables():
    """Drop and recreate all database tables (for development only!)"""
    with db:
        # db.drop_tables([Gender, Teacher, PhoneNumber, Student, AnalysisLog, ActivityLog, Notification], safe=True)
        db.create_tables([Gender, Teacher, PhoneNumber, Student, AnalysisLog, ActivityLog, Notification,DownloadLog], safe=True)
        print("All tables created successfully.")


if __name__ == '__main__':
    create_tables()




