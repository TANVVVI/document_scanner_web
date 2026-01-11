# 📄 Document Scanner Web Application

A web-based **automatic document scanner** that detects documents from images,
removes perspective distortion, shadows, and background noise, and produces
a clean scanned output similar to CamScanner or Adobe Scan.

---

## 🚀 Features

- Automatic document detection
- Perspective correction using homography
- Shadow and background removal
- Clean scanned output
- Flask-based web interface
- Modern gradient UI
- Download scanned document

---

## 🧠 How It Works

1. Upload a document image
2. Image preprocessing and enhancement
3. Document boundary detection using OpenCV
4. Perspective transformation (homography)
5. Shadow removal and background cleaning
6. Final scanned document output

---

## 🖥️ Tech Stack

- Python
- OpenCV
- NumPy
- Flask
- HTML & CSS
- Computer Vision

---

## 📂 Project Structure

document_scanner_web/
├── app.py
├── config.py
├── scanner/
│ └── scanner.py
├── templates/
│ └── index.html
├── static/
│ ├── css/
│ │ └── style.css
│ ├── uploads/
│ └── outputs/
├── requirements.txt
└── README.md


---

## ⚙️ How to Run Locally

```bash
pip install -r requirements.txt
python app.py


http://127.0.0.1:5000


🎓 Project Value

This project demonstrates:

Real-world computer vision pipeline

Image preprocessing and geometry

Homography-based document scanning

Flask backend integration

Clean UI design

👤 Author

Tanvi Salvi
Python & Computer Vision Developer




