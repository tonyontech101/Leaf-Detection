# 🌿 Leaf Detection

A local-first web application that identifies plant species and assesses leaf health using deep learning. Upload a photo of a leaf and the app will predict its species via kNN similarity search and optionally analyze its condition using a local vision-language model (VLM).

## Features

- **Species Identification** — Uses a ConvNeXt backbone to extract embeddings and kNN to classify across 22 leaf classes (8 species × various conditions).
- **Health Analysis** — Optionally uses a local Ollama VLM (e.g. Moondream) for free-text condition descriptions, combined with OpenCV colour heuristics for an overall health status.
- **Similar Image Gallery** — Returns the most visually similar leaves from the dataset.
- **Fully Offline** — After initial setup, the app runs entirely on localhost with no internet required.
- **PWA Support** — Installable as a Progressive Web App with a service worker for offline caching.

## Supported Species & Conditions

| Species | Conditions |
|---|---|
| Aloe Vera | Disease, Dried, Mature Healthy, Young Healthy |
| Azadirachta Indica | Chlorotic, Disease, Healthy |
| Centella Asiatica | Healthy, Insects, Mild Disease |
| Hibiscus Rosa Sinensis | Chlorotic, Disease, Healthy |
| Kalanchoe Pinnata | Chlorotic, Disease, Healthy |
| Mikania Micrantha | Disease, Distorted, Healthy |
| Piper Betle | Chlorotic, Disease, Healthy |

## Project Structure

```
LeafDetection/
├── run.py                     # Entry point — launches the app
├── requirements.txt           # Python dependencies
├── Original Dataset/          # ⬅ Place the dataset here (see below)
│   ├── Aloe Vera Disease/
│   ├── Aloe Vera Dried/
│   ├── ...
│   └── Piper Betle Healthy/
└── app/
    ├── artifacts/             # Generated index, labels, thumbnails, model cache
    │   ├── index.npz
    │   ├── labels.json
    │   ├── manifest.json
    │   ├── thumbnails/
    │   └── torch_cache/
    ├── backend/               # FastAPI backend
    │   ├── config.py          # Central configuration & paths
    │   ├── main.py            # API routes & static file serving
    │   ├── embedding.py       # Feature extraction
    │   ├── inference.py       # kNN species prediction
    │   ├── health.py          # Leaf health analysis (VLM + OpenCV)
    │   ├── leaf_utils.py      # Image processing utilities
    │   └── plant_info.py      # Plant species information
    ├── frontend/              # Static HTML/CSS/JS frontend
    │   ├── index.html
    │   ├── styles.css
    │   ├── app.js
    │   └── sw.js              # Service worker
    └── scripts/               # Setup & preprocessing scripts
        ├── setup_offline.py
        ├── preprocess_build_index.py
        ├── evaluate_accuracy.py
        └── verify_offline.py
```

## ⚠️ Database / Dataset Placement

The image dataset must be placed in the **`Original Dataset/`** folder at the **project root** (i.e. the same level as `run.py`).

```
LeafDetection/
├── run.py
├── Original Dataset/       ⬅  Place the dataset folder here
│   ├── Aloe Vera Disease/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   ├── Aloe Vera Dried/
│   └── ... (22 class folders total)
```

Each subfolder name should match a class label (species + condition, e.g. `Piper Betle Healthy`) and contain the corresponding leaf images. The dataset is **not included in the repository** — you must obtain or provide it separately.

> **Note:** The `Original Dataset/` folder is listed in `.gitignore` and will not be committed to version control.

## Getting Started

### Prerequisites

- Python 3.10+
- (Optional) [Ollama](https://ollama.com/) with the `moondream` model for health analysis

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tonyontech101/Leaf-Detection.git
   cd Leaf-Detection
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place the dataset** in the `Original Dataset/` folder (see [Database / Dataset Placement](#️-database--dataset-placement) above).

5. **Run setup & build the index**
   ```bash
   python -m app.scripts.setup_offline
   python -m app.scripts.preprocess_build_index
   ```

6. **Verify offline readiness** *(optional)*
   ```bash
   python -m app.scripts.verify_offline
   ```

### Running the App

```bash
python run.py
```

The app will launch at **http://127.0.0.1:8000** and automatically open in your default browser.

## Configuration

Configuration is managed in [`app/backend/config.py`](app/backend/config.py). Key settings can be overridden via environment variables:

| Environment Variable | Default | Description |
|---|---|---|
| `LEAF_EMBEDDING_MODEL` | `convnext_small` | Backbone model (`mobilenet_v3_large`, `convnext_tiny`, `convnext_small`, `efficientnet_v2_s`) |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL for VLM health analysis |
| `LEAF_VLM_MODEL` | `moondream` | Vision-language model name |

> **Important:** Changing the embedding model requires re-running setup and rebuilding the index.
