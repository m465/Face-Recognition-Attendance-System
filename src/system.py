import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from PIL import Image
import numpy as np
import os
import faiss
import pandas as pd
import json
from datetime import datetime, timedelta

class FaissAttendanceSystem:
    def __init__(self, dataset_path, attendance_file, model_cache_dir="models"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 1. Models
        self.mtcnn = MTCNN(image_size=160, margin=14, device=self.device, post_process=True)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # 2. Database Paths
        self.dataset_path = dataset_path
        self.attendance_file = attendance_file
        self.model_cache_dir = model_cache_dir
        self.index_path = os.path.join(model_cache_dir, "index.faiss")
        self.metadata_path = os.path.join(model_cache_dir, "metadata.json")
        
        if not os.path.exists(self.model_cache_dir):
            os.makedirs(self.model_cache_dir)

        # 3. Vector DB State
        self.embedding_dim = 512
        self.index = None
        self.index_to_name = {}
        
        # 4. Attendance Config
        self.cooldown_dict = {} 
        self.buffer_seconds = 10 
        
        # Initialize Database (Load or Build)
        self._initialize_system()

    def get_embedding(self, face_tensor):
        with torch.no_grad():
            if face_tensor.shape[0] == 1: face_tensor = face_tensor.repeat(3, 1, 1)
            embedding = self.resnet(face_tensor.unsqueeze(0).to(self.device))
            embedding = embedding.cpu().numpy().astype('float32')
            faiss.normalize_L2(embedding)
            return embedding

    def _get_dataset_fingerprint(self):
        """Creates a snapshot of folders and file counts to detect changes."""
        fingerprint = {}
        for person_name in os.listdir(self.dataset_path):
            person_dir = os.path.join(self.dataset_path, person_name)
            if os.path.isdir(person_dir):
                files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                fingerprint[person_name] = len(files)
        return fingerprint

    def _initialize_system(self):
        """Decides whether to load existing index or rebuild from scratch."""
        current_fingerprint = self._get_dataset_fingerprint()
        
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if the dataset has changed
            if metadata.get("fingerprint") == current_fingerprint:
                print("🚀 No changes detected in dataset. Loading cached FAISS index...")
                self.index = faiss.read_index(self.index_path)
                # Convert keys back to int because JSON makes them strings
                self.index_to_name = {int(k): v for k, v in metadata["names"].items()}
                return

        print("🔄 Dataset changed or cache missing. Rebuilding embeddings (this may take a while)...")
        self._build_database(current_fingerprint)

    def _build_database(self, fingerprint):
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index_to_name = {}
        current_id = 0
        
        for person_name in fingerprint.keys():
            person_dir = os.path.join(self.dataset_path, person_name)
            person_embeddings = []
            
            for img_name in os.listdir(person_dir):
                try:
                    img = Image.open(os.path.join(person_dir, img_name)).convert('RGB')
                    face_tensor = self.mtcnn(img)
                    if face_tensor is not None:
                        person_embeddings.append(self.get_embedding(face_tensor))
                except: continue

            if person_embeddings:
                avg_emb = np.mean(np.vstack(person_embeddings), axis=0, keepdims=True)
                self.index.add(avg_emb)
                self.index_to_name[current_id] = person_name
                current_id += 1

        # Save to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump({
                "fingerprint": fingerprint,
                "names": self.index_to_name
            }, f)
        print(f"✅ Database built and saved with {current_id} persons.")

    def mark_attendance(self, name):
        now = datetime.now()
        date_today = now.strftime("%Y-%m-%d")
        time_now = now.strftime("%H:%M:%S")
        
        if name in self.cooldown_dict:
            if now < self.cooldown_dict[name] + timedelta(seconds=self.buffer_seconds):
                return

        if os.path.exists(self.attendance_file) and os.path.getsize(self.attendance_file) > 0:
            df = pd.read_csv(self.attendance_file)
        else:
            df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])

        mask = (df['Name'] == name) & (df['Date'] == date_today)
        
        if mask.any():
            df.loc[mask, 'Status'] = "Duplicate"
            df.loc[mask, 'Time'] = time_now
        else:
            new_row = {"Name": name, "Date": date_today, "Time": time_now, "Status": "Original"}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(self.attendance_file, index=False)
        self.cooldown_dict[name] = now

    def run_webcam(self, threshold=0.75):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: break

            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, _ = self.mtcnn.detect(img_pil)
            faces = self.mtcnn(img_pil)

            if boxes is not None and faces is not None:
                if torch.is_tensor(faces): faces = [faces]
                for box, face in zip(boxes, faces):
                    emb = self.get_embedding(face)
                    distances, indices = self.index.search(emb, k=1)
                    dist, idx = distances[0][0], indices[0][0]

                    if idx != -1 and dist < threshold:
                        name = self.index_to_name[idx]
                        self.mark_attendance(name)
                        color = (0, 255, 0)
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)

                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow('Attendance System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()