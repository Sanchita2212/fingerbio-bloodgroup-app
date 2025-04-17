from app import app

if __name__ == '__main__':
    app.secret_key = 'sanchita'  # Set a secure secret key
    app.run(debug=True)

