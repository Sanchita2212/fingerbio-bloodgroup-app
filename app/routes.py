from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import os
from app import app
from app.model_loader import load_model, predict_image

# Load model once
model = None
try:
    model = load_model(os.path.join("model", "blood_group_model_best.pth"))
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/about')  # ✅ Added this to fix the BuildError
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    # your register logic here
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Add your login logic here (e.g., authentication)
        username = request.form.get('username')
        password = request.form.get('password')
        # Example: Validate username and password
        if username == 'admin' and password == 'password':
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials. Please try again.', 'error')
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash('Model not loaded properly, try again later.', 'error')
        return redirect(url_for('upload'))

    if 'image' not in request.files:
        flash('No file uploaded.', 'error')
        return redirect(url_for('upload'))

    file = request.files['image']
    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('upload'))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('app/static/uploads', filename)
        file.save(filepath)

        # ✅ Use your CNN model to predict
        blood_group = predict_image(model, filepath)

        session['prediction'] = blood_group
        session['image_path'] = filename
        return redirect(url_for('result'))

    flash('Something went wrong.', 'error')
    return redirect(url_for('upload'))

@app.route('/result')
def result():
    prediction = session.get('prediction', 'Unknown')
    image_path = session.get('image_path', None)
    return render_template('result.html', prediction=prediction, image_path=image_path)

@app.route('/logout')
def logout():
    # your logout logic here (e.g., session pop, redirect)
    return redirect(url_for('home'))  # or wherever you want to redirect after logout


