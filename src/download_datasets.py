import os
import subprocess
from datasets import load_dataset, Dataset

DATA_DIR = "datasets"
os.makedirs(DATA_DIR, exist_ok=True)

def download_hf_dataset(repo_id, name, subset_size=None, split="train"):
    print(f"\n--- Downloading {repo_id} ---")
    local_path = os.path.join(DATA_DIR, name)
    if os.path.exists(local_path):
        print(f"Dataset {name} already exists at {local_path}. Skipping.")
        return

    try:
        if subset_size:
            print(f"Downloading subset of {subset_size} samples (streaming mode)...")
            ds = load_dataset(repo_id, streaming=True, split=split)
            samples = list(ds.take(subset_size))
            saved_ds = Dataset.from_list(samples)
            saved_ds.save_to_disk(local_path)
        else:
            print("Downloading full dataset...")
            ds = load_dataset(repo_id, split=split)
            ds.save_to_disk(local_path)
        print(f"Saved to {local_path}")
    except Exception as e:
        print(f"Failed to download {repo_id}: {e}")

def download_kaggle_dataset(dataset_id, name):
    print(f"\n--- Downloading Kaggle Dataset: {dataset_id} ---")
    local_path = os.path.join(DATA_DIR, name)
    if os.path.exists(local_path) and os.listdir(local_path):
        print(f"Dataset {name} already exists at {local_path}. Skipping.")
        return
        
    os.makedirs(local_path, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download", 
        "-d", dataset_id, 
        "-p", local_path, 
        "--unzip"
    ]
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"Saved to {local_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download Kaggle dataset: {e}")

if __name__ == "__main__":
    print("Starting dataset downloads...\n")
    
    # Text datasets
    download_hf_dataset("hungsvdut2k2/vietnamese-medical-chat-data", "vietnamese_medical_chat", split="train")
    
    # Audio datasets (Vietnamese)
    download_hf_dataset("tensorxt/ViMedCSS", "vimedcss", split="train")
    
    # Download a small subset of VietMed (500 samples) to avoid excessive disk usage
    download_hf_dataset("leduckhai/VietMed", "vietmed_subset", subset_size=500, split="test")
    
    # Audio datasets (English)
    download_hf_dataset("ekacare/eka-medical-asr-evaluation-dataset", "eka_medical_asr", split="test")
    
    # Kaggle Audio dataset
    download_kaggle_dataset("paultimothymooney/medical-speech-transcription-and-intent", "kaggle_medical_speech")
    
    print("\n--- All downloads completed! ---")
