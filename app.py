#!/usr/bin/env python3
"""
Document Scanner Web Application - Flask Backend
"""

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime

from scanner.scanner import DocumentScanner
from config import Config

# --------------------------------------------------
# APP SETUP
# --------------------------------------------------

app = Flask(__name__)
app.config.from_object(Config)

scanner = DocumentScanner()

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def generate_unique_filename(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    uid = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{timestamp}_{uid}.{ext}"

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    original = None
    scanned = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            error = "No file selected."
            return render_template("index.html", error=error)

        if not allowed_file(file.filename):
            error = "Invalid file type."
            return render_template("index.html", error=error)

        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_name = generate_unique_filename(filename)

        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        output_name = f"scanned_{unique_name}"
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_name)

        file.save(upload_path)

        # Scan document
        result = scanner.scan_document(
            input_path=upload_path,
            output_path=output_path,
            enhance_mode="adaptive",
            width=app.config["PROCESSING_WIDTH"]
        )

        if not result["success"]:
            error = result["message"]
            return render_template("index.html", error=error)

        original = url_for("static", filename=f"uploads/{unique_name}")
        scanned = url_for("static", filename=f"outputs/{output_name}")

    return render_template(
        "index.html",
        original=original,
        scanned=scanned,
        error=error
    )

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=app.config["DEBUG"]
    )
