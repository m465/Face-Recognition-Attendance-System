# Face Recognition Attendance System (FAISS-Based)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-VectorDB-green.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A professional, modularized attendance system that uses deep learning to identify faces in real-time and log attendance into a CSV file. The system is optimized for speed using **FAISS (Facebook AI Similarity Search)** and features an intelligent re-indexing logic to handle large datasets efficiently.

## 🚀 Key Features

*   **Real-time Detection:** Uses MTCNN for high-accuracy face detection and alignment.
*   **High-Fidelity Embeddings:** Leverages `InceptionResnetV1` (pretrained on VGGFace2) to generate 512-dimensional face vectors.
*   **Ultra-fast Search:** Implements FAISS (Vector Database) for sub-millisecond face matching, even with thousands of registered users.
*   **Smart Caching:** Automatically saves embeddings to a local index. It only re-indexes the dataset if it detects changes (new images or new folders) using a directory fingerprinting system.
*   **Robust Logging:** Generates daily CSV reports categorized by "Original" (first check-in) and "Duplicate" (subsequent check-ins).
*   **Optimized Performance:** Supports GPU acceleration (CUDA) for both face detection and embedding generation.

---

## 🛠️ Project Structure

```text
attendance-system/
│
├── main.py                 # Entry point: Initializes and runs the webcam system
├── requirements.txt        # Python dependencies
│
├── src/                    # Source Code
│   ├── __init__.py         # Package initialization
│   ├── system.py           # Core logic (Face Recognition & FAISS)
│   └── utils.py            # Helper functions (Directory management)
│
├── dataset/                # Face Data (User folders with images)
│   └── John_Doe/           # Folder name = Person's Name
│       ├── img1.jpg
│       └── img2.jpg
│
├── models/                 # Persistent storage for Vector DB
│   ├── index.faiss         # Cached face vectors
│   └── metadata.json       # Mapping of IDs to Names & Dataset fingerprint
│
└── logs/                   # Attendance outputs
    └── attendance_log.csv  # Auto-generated attendance reports
```

---

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/attendance-system.git
    cd attendance-system
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you have an NVIDIA GPU, ensure you have the correct version of `faiss-gpu` and `torch` with CUDA support.*

---

## 📖 Usage

### 1. Prepare the Dataset
Create a folder for each person inside the `dataset/` directory. Place 2-5 clear images of their face inside their respective folder.
```text
dataset/
├── Elon_Musk/
│   ├── musk1.jpg
│   └── musk2.jpg
└── Jeff_Bezos/
    └── bezos1.jpg
```

### 2. Run the System
Execute the main script to start the webcam:
```bash
python main.py
```

*   The system will scan the `dataset` folder.
*   If it is the first run, it will generate embeddings and save them to `models/`.
*   On subsequent runs, it will load the cached database in milliseconds.
*   Press **'q'** to exit the webcam view.

---

## 🔍 How it Works

1.  **Detection:** MTCNN locates faces in the video frame and crops them.
2.  **Feature Extraction:** The cropped face is passed through InceptionResnetV1 to produce a 512-unit vector (embedding).
3.  **Matching:** FAISS compares the real-time embedding against the stored database using L2 (Euclidean) distance.
4.  **Verification:** If the distance is below the `threshold` (default 0.70), the face is identified.
5.  **Logging:** The system checks the `attendance_log.csv`. If the person hasn't checked in today, it marks them as "Original." If they were already seen, it updates their status to "Duplicate."

---

## 🛠️ Technologies Used

*   **Python:** Core programming.
*   **PyTorch:** Deep learning framework for MTCNN and ResNet.
*   **OpenCV:** Video stream processing and UI visualization.
*   **FAISS:** Facebook AI Similarity Search for high-speed vector indexing.
*   **Pandas:** Data management for attendance logging.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
