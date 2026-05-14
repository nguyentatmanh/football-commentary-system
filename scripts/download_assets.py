import os
import sys

def download():
    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Installing dependencies first...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    # Set paths
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(ROOT, "data", "models")
    RAW_DIR = os.path.join(ROOT, "data", "raw")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    # List of items to download (id, output_path)
    downloads = [
        # Models
        ("1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V", os.path.join(MODELS_DIR, "football-ball-detection.pt")),
        ("17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q", os.path.join(MODELS_DIR, "football-player-detection.pt")),
        ("1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf", os.path.join(MODELS_DIR, "football-pitch-detection.pt")),
        
        # Default sample video
        ("12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF", os.path.join(RAW_DIR, "sample.mp4")),
    ]

    print("Downloading football analytics assets...")
    for file_id, dest in downloads:
        if os.path.exists(dest):
            print(f"Skipping already existing: {os.path.basename(dest)}")
            continue
            
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {os.path.basename(dest)}...")
        gdown.download(url, dest, quiet=False)
        
    print("All assets downloaded!")

if __name__ == "__main__":
    download()
