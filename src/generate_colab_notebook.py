import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Onevoice Healthcare Solution: Data Pipeline & Advanced EDA\n",
                "Notebook này được thiết kế để chạy trên **Google Colab Pro** nhằm tải toàn bộ dữ liệu âm thanh và văn bản y tế có dung lượng lớn, sau đó thực hiện Khám phá dữ liệu (EDA) từ cơ bản đến nâng cao.\n",
                "\n",
                "## Hướng dẫn Kaggle\n",
                "Hãy đảm bảo bạn đã upload file `kaggle.json` vào Colab (hoặc mount Google Drive) để tải dataset Kaggle."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q datasets huggingface_hub kaggle librosa matplotlib seaborn soundfile pandas wordcloud nltk"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "# Thiết lập Kaggle API (Yêu cầu upload file kaggle.json)\n",
                "if not os.path.exists('/root/.kaggle/kaggle.json'):\n",
                "    !mkdir -p ~/.kaggle\n",
                "    print('Vui lòng upload file kaggle.json vào thư mục hiện tại.')\n",
                "    from google.colab import files\n",
                "    uploaded = files.upload()\n",
                "    for fn in uploaded.keys():\n",
                "        !cp {fn} ~/.kaggle/kaggle.json\n",
                "    !chmod 600 ~/.kaggle/kaggle.json"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Tải Dữ liệu (Data Downloading)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import subprocess\n",
                "from datasets import load_dataset\n",
                "\n",
                "DATA_DIR = \"/content/datasets\"\n",
                "os.makedirs(DATA_DIR, exist_ok=True)\n",
                "\n",
                "def download_hf_dataset(repo_id, name, split=\"train\"):\n",
                "    print(f\"\\n--- Tải {repo_id} ---\")\n",
                "    local_path = os.path.join(DATA_DIR, name)\n",
                "    ds = load_dataset(repo_id, split=split)\n",
                "    ds.save_to_disk(local_path)\n",
                "    print(f\"Đã lưu tại {local_path}\")\n",
                "\n",
                "def download_kaggle_dataset(dataset_id, name):\n",
                "    print(f\"\\n--- Tải Kaggle: {dataset_id} ---\")\n",
                "    local_path = os.path.join(DATA_DIR, name)\n",
                "    os.makedirs(local_path, exist_ok=True)\n",
                "    !kaggle datasets download -d {dataset_id} -p {local_path} --unzip\n",
                "    print(f\"Đã lưu tại {local_path}\")\n",
                "\n",
                "print(\"Bắt đầu kéo dữ liệu... (Tải full dataset)\")\n",
                "download_hf_dataset(\"hungsvdut2k2/vietnamese-medical-chat-data\", \"vietnamese_medical_chat\", split=\"train\")\n",
                "download_hf_dataset(\"tensorxt/ViMedCSS\", \"vimedcss\", split=\"train\")\n",
                "download_hf_dataset(\"leduckhai/VietMed\", \"vietmed_full\", split=\"test\")\n",
                "download_hf_dataset(\"ekacare/eka-medical-asr-evaluation-dataset\", \"eka_medical_asr\", split=\"test\")\n",
                "download_kaggle_dataset(\"paultimothymooney/medical-speech-transcription-and-intent\", \"kaggle_medical_speech\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Phân tích Khám phá Nâng cao - Text (Advanced Text EDA)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from wordcloud import WordCloud\n",
                "from datasets import load_from_disk\n",
                "\n",
                "print(\"=== Advanced EDA: Text Dataset ===\")\n",
                "ds_text = load_from_disk(f\"{DATA_DIR}/vietnamese_medical_chat\")\n",
                "df_text = ds_text.to_pandas()\n",
                "\n",
                "# Tìm cột chứa Text\n",
                "text_cols = [c for c in df_text.columns if df_text[c].dtype == 'object'][:2]\n",
                "\n",
                "# Phân tích phân phối độ dài & Dữ liệu rỗng\n",
                "for col in text_cols:\n",
                "    df_text[f'{col}_len'] = df_text[col].astype(str).apply(lambda x: len(x.split()))\n",
                "    empty_count = df_text[col].isnull().sum() + (df_text[col] == '').sum()\n",
                "    print(f\"Cột '{col}': {empty_count} mẫu bị rỗng/thiếu.\")\n",
                "\n",
                "plt.figure(figsize=(10, 5))\n",
                "for col in text_cols:\n",
                "    sns.histplot(df_text[f'{col}_len'], kde=True, label=col, bins=50)\n",
                "plt.title(\"Phân phối độ dài câu hội thoại (Word Count)\")\n",
                "plt.xlabel(\"Số lượng từ\")\n",
                "plt.legend()\n",
                "plt.show()\n",
                "\n",
                "# Phân tích từ vựng phổ biến (WordCloud)\n",
                "if text_cols:\n",
                "    all_text = \" \".join(df_text[text_cols[0]].astype(str).tolist())\n",
                "    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)\n",
                "    plt.figure(figsize=(10, 5))\n",
                "    plt.imshow(wordcloud, interpolation='bilinear')\n",
                "    plt.axis('off')\n",
                "    plt.title(f\"WordCloud - Tần suất từ vựng trong cột {text_cols[0]}\")\n",
                "    plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Phân tích Khám phá Nâng cao - Audio (Advanced Audio EDA)\n",
                "Vẽ sóng âm (Waveform), Spectrogram, tính toán tỷ lệ Sample Rate và Tốc độ nói (Characters per second)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import librosa\n",
                "import librosa.display\n",
                "import IPython.display as ipd\n",
                "\n",
                "def advanced_audio_eda(path, name, sample_limit=500):\n",
                "    print(f\"\\n=== Advanced EDA Audio: {name} ===\")\n",
                "    try:\n",
                "        ds = load_from_disk(path)\n",
                "        \n",
                "        audio_col = next((c for c, f in ds.features.items() if hasattr(f, 'sampling_rate') or 'audio' in c.lower()), None)\n",
                "        text_col = next((c for c in ds.features.keys() if 'text' in c.lower() or 'transcript' in c.lower()), None)\n",
                "        if not audio_col: return\n",
                "        \n",
                "        durations = []\n",
                "        sample_rates = []\n",
                "        speech_rates = []\n",
                "        \n",
                "        sample_size = min(sample_limit, len(ds))\n",
                "        for i in range(sample_size):\n",
                "            item = ds[i][audio_col]\n",
                "            if isinstance(item, dict) and 'array' in item and 'sampling_rate' in item:\n",
                "                dur = len(item['array']) / item['sampling_rate']\n",
                "                durations.append(dur)\n",
                "                sample_rates.append(item['sampling_rate'])\n",
                "                \n",
                "                if text_col and ds[i][text_col]:\n",
                "                    char_count = len(str(ds[i][text_col]))\n",
                "                    if dur > 0:\n",
                "                        speech_rates.append(char_count / dur)\n",
                "\n",
                "        # 1. Phân phối độ dài Audio & Sample Rate\n",
                "        fig, axes = plt.subplots(1, 2, figsize=(15, 4))\n",
                "        sns.histplot(durations, kde=True, bins=50, color='coral', ax=axes[0])\n",
                "        axes[0].set_title(\"Audio Duration (s)\")\n",
                "        \n",
                "        sns.countplot(x=sample_rates, ax=axes[1], palette='viridis')\n",
                "        axes[1].set_title(\"Sample Rate Distribution (Hz)\")\n",
                "        plt.show()\n",
                "        \n",
                "        # 2. Phân phối tốc độ nói (Characters per second)\n",
                "        if speech_rates:\n",
                "            plt.figure(figsize=(8, 4))\n",
                "            sns.histplot(speech_rates, kde=True, bins=50, color='purple')\n",
                "            plt.title(\"Tốc độ nói (Characters / Second)\")\n",
                "            plt.show()\n",
                "            \n",
                "        # 3. Phân tích Tín hiệu chuyên sâu (Waveform & Spectrogram) trên 1 mẫu ngẫu nhiên\n",
                "        print(\"\\n--- Phân tích Mẫu ngẫu nhiên (Spectrogram & Waveform) ---\")\n",
                "        sample_audio = ds[0][audio_col]['array']\n",
                "        sample_sr = ds[0][audio_col]['sampling_rate']\n",
                "        \n",
                "        fig, ax = plt.subplots(2, 1, figsize=(12, 8))\n",
                "        # Waveform\n",
                "        librosa.display.waveshow(sample_audio, sr=sample_sr, ax=ax[0])\n",
                "        ax[0].set_title(\"Waveform\")\n",
                "        \n",
                "        # Spectrogram\n",
                "        D = librosa.amplitude_to_db(np.abs(librosa.stft(sample_audio)), ref=np.max)\n",
                "        img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sample_sr, ax=ax[1])\n",
                "        fig.colorbar(img, ax=ax[1], format=\"%+2.0f dB\")\n",
                "        ax[1].set_title(\"Linear-frequency power spectrogram\")\n",
                "        plt.tight_layout()\n",
                "        plt.show()\n",
                "        \n",
                "    except Exception as e:\n",
                "        print(\"Lỗi:\", e)\n",
                "\n",
                "advanced_audio_eda(f\"{DATA_DIR}/vimedcss\", \"ViMedCSS (Code-switching)\")\n",
                "advanced_audio_eda(f\"{DATA_DIR}/vietmed_full\", \"VietMed (Vietnamese ASR)\")\n",
                "advanced_audio_eda(f\"{DATA_DIR}/eka_medical_asr\", \"Eka ASR (English)\")\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("Onevoice_Colab_EDA.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("Đã tạo file Onevoice_Colab_EDA.ipynb với Advanced EDA thành công!")
