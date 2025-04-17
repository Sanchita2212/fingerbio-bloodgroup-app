from flask import Flask

app = Flask(__name__)
app.secret_key = 'sanchita'  # Needed for sessions (login, register, etc.)

from app import routes  # Import routes after app is created
