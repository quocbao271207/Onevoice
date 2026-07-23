import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
import librosa
from pathlib import Path

DATA_DIR = "datasets"
PLOTS_DIR = os.path.join(DATA_DIR, "eda_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

report_content = ["# Medical Datasets EDA Report\n\n"]

def write_to_report(text):
    print(text)
    report_content.append(text + "\n")

def eda_text_dataset(dataset_path, name):
    write_to_report(f"## 1. Text Dataset EDA: {name}")
    if not os.path.exists(dataset_path):
        write_to_report(f"**Error:** Dataset not found at {dataset_path}")
        return
    
    try:
        ds = load_from_disk(dataset_path)
        write_to_report(f"- **Total samples:** {len(ds)}")
        write_to_report(f"- **Features:** {list(ds.features.keys())}")
        
        # Analyze lengths
        df = ds.to_pandas()
        
        # Determine the text columns based on dataset name
        text_cols = []
        if 'Question' in df.columns and 'Answer' in df.columns:
            text_cols = ['Question', 'Answer']
        elif 'input' in df.columns and 'output' in df.columns:
            text_cols = ['input', 'output']
        else:
            text_cols = [c for c in df.columns if df[c].dtype == 'object'][:2]
            
        for col in text_cols:
            df[f'{col}_len'] = df[col].astype(str).apply(lambda x: len(x.split()))
            
        if text_cols:
            plt.figure(figsize=(10, 5))
            for col in text_cols:
                sns.histplot(df[f'{col}_len'], kde=True, label=col, bins=50)
            plt.title(f"{name} - Word Count Distribution")
            plt.legend()
            plot_path = os.path.join(PLOTS_DIR, f"{name}_lengths.png")
            plt.savefig(plot_path)
            plt.close()
            write_to_report(f"- **Word Count Distribution:** Saved to `{plot_path}`")
            write_to_report(f"- **Observations:** Text column lengths vary. Check {plot_path} for skewness.")
            write_to_report(f"- **Next steps:** Filter out extremely short (e.g., < 3 words) or extremely long responses to maintain fine-tuning quality.\n")
    except Exception as e:
        write_to_report(f"**Error analyzing {name}:** {e}")

def eda_audio_hf_dataset(dataset_path, name):
    write_to_report(f"## 2. Audio Dataset EDA (HuggingFace): {name}")
    if not os.path.exists(dataset_path):
        write_to_report(f"**Error:** Dataset not found at {dataset_path}")
        return
        
    try:
        ds = load_from_disk(dataset_path)
        write_to_report(f"- **Total samples:** {len(ds)}")
        write_to_report(f"- **Features:** {list(ds.features.keys())}")
        
        # We will sample max 500 for audio duration to save time
        sample_size = min(500, len(ds))
        durations = []
        sample_rates = []
        
        # Identify audio column
        audio_col = None
        for col, feature in ds.features.items():
            if hasattr(feature, 'sampling_rate') or 'audio' in col.lower():
                audio_col = col
                break
                
        if not audio_col:
            write_to_report("**Error:** Could not identify audio column.")
            return
            
        for i in range(sample_size):
            item = ds[i][audio_col]
            if isinstance(item, dict) and 'array' in item and 'sampling_rate' in item:
                sr = item['sampling_rate']
                dur = len(item['array']) / sr
                durations.append(dur)
                sample_rates.append(sr)
        
        if durations:
            plt.figure(figsize=(10, 5))
            sns.histplot(durations, kde=True, bins=50, color='coral')
            plt.title(f"{name} - Audio Duration Distribution (seconds)")
            plt.xlabel("Duration (s)")
            plot_path = os.path.join(PLOTS_DIR, f"{name}_durations.png")
            plt.savefig(plot_path)
            plt.close()
            
            avg_dur = sum(durations)/len(durations)
            unique_srs = set(sample_rates)
            
            write_to_report(f"- **Average Duration:** {avg_dur:.2f} seconds")
            write_to_report(f"- **Sample Rates Detected:** {unique_srs}")
            write_to_report(f"- **Duration Plot:** Saved to `{plot_path}`")
            write_to_report(f"- **Next steps:** Resample all audio to 16kHz for Whisper. Filter out clips > 30s or < 1s to prevent out-of-memory errors and noise.\n")
    except Exception as e:
        write_to_report(f"**Error analyzing {name}:** {e}")

def eda_kaggle_audio(dataset_path, name):
    write_to_report(f"## 3. Audio Dataset EDA (Kaggle Folders): {name}")
    if not os.path.exists(dataset_path):
        write_to_report(f"**Error:** Dataset not found at {dataset_path}")
        return
        
    wav_files = list(Path(dataset_path).rglob("*.wav"))
    write_to_report(f"- **Total .wav files:** {len(wav_files)}")
    
    if not wav_files:
        return
        
    sample_size = min(500, len(wav_files))
    durations = []
    
    # Process a subset to be fast
    for f in wav_files[:sample_size]:
        try:
            dur = librosa.get_duration(path=f)
            durations.append(dur)
        except Exception:
            pass
            
    if durations:
        plt.figure(figsize=(10, 5))
        sns.histplot(durations, kde=True, bins=50, color='green')
        plt.title(f"{name} - Audio Duration Distribution (seconds)")
        plt.xlabel("Duration (s)")
        plot_path = os.path.join(PLOTS_DIR, f"{name}_durations.png")
        plt.savefig(plot_path)
        plt.close()
        
        avg_dur = sum(durations)/len(durations)
        write_to_report(f"- **Average Duration:** {avg_dur:.2f} seconds")
        write_to_report(f"- **Duration Plot:** Saved to `{plot_path}`")
        write_to_report(f"- **Next steps:** Convert Kaggle folder structure into a unified huggingface Dataset format with transcripts mapped from CSVs.\n")

if __name__ == "__main__":
    print("Starting EDA...")
    
    eda_text_dataset(os.path.join(DATA_DIR, "vietnamese_medical_chat"), "Vietnamese_Medical_Chat")
    eda_audio_hf_dataset(os.path.join(DATA_DIR, "vimedcss"), "ViMedCSS")
    eda_audio_hf_dataset(os.path.join(DATA_DIR, "vietmed_subset"), "VietMed_Subset")
    eda_audio_hf_dataset(os.path.join(DATA_DIR, "eka_medical_asr"), "Eka_Medical_ASR")
    eda_kaggle_audio(os.path.join(DATA_DIR, "kaggle_medical_speech"), "Kaggle_Medical_Speech")
    
    with open(os.path.join(DATA_DIR, "eda_report.md"), "w") as f:
        f.writelines(report_content)
        
    print(f"EDA complete. Report saved to {os.path.join(DATA_DIR, 'eda_report.md')}")
