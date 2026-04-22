import os
from src.system import FaissAttendanceSystem
from src.utils import create_dirs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
LOGS_PATH = os.path.join(BASE_DIR, "logs")
MODELS_PATH = os.path.join(BASE_DIR, "models") # New
ATTENDANCE_FILE = os.path.join(LOGS_PATH, "attendance_log.csv")

def main():
    create_dirs([DATASET_PATH, LOGS_PATH, MODELS_PATH])

    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        print(f"❌ Dataset empty: {DATASET_PATH}")
        return

    system = FaissAttendanceSystem(
        dataset_path=DATASET_PATH, 
        attendance_file=ATTENDANCE_FILE,
        model_cache_dir=MODELS_PATH
    )
    
    system.run_webcam(threshold=0.70)

if __name__ == "__main__":
    main()