# 🌿 Leaf Detection

A local-first web application that identifies plant species and assesses leaf health using deep learning. Upload a photo of a leaf and the app will predict its species via kNN similarity search and optionally analyze its condition using a local vision-language model (VLM).

## Features

- **Accounts & Login** — Local sign-up / sign-in with **two modes**: email + password, or **face recognition** using face embeddings. All accounts and face data stay on your machine.
- **Species Identification** — Uses a ConvNeXt backbone to extract embeddings and kNN to classify across 22 leaf classes (7 species × various conditions).
- **Health Analysis** — Combines a local Ollama VLM (e.g. Moondream) with OpenCV colour heuristics to report an overall health status and structured **symptoms, treatment, and prevention** guidance.
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

The reference dataset used to build the index contains **1,981 images** across these 22 classes.

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
    │   ├── auth_db.py         # Local accounts store (SQLite) + password hashing + sessions
    │   ├── auth_routes.py     # Auth API: signup / login / face-login / logout
    │   ├── face_auth.py       # Face detect + crop + embed + match (offline)
    │   ├── embedding.py       # Feature extraction
    │   ├── inference.py       # kNN species prediction
    │   ├── health.py          # Leaf health analysis (VLM + OpenCV)
    │   ├── leaf_utils.py      # Image processing utilities
    │   └── plant_info.py      # Plant species information
    ├── frontend/              # Static HTML/CSS/JS frontend
    │   ├── index.html         # Main app (auth-gated)
    │   ├── login.html         # Sign in / create account page
    │   ├── styles.css
    │   ├── app.js
    │   ├── auth.js            # Login / signup / face-login logic
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

## Health Analysis with Ollama (optional)

Species identification, similarity search, and the OpenCV colour heuristic all
work **without** Ollama. To enable the richer AI health analysis (structured
symptoms, treatment, and prevention), run a local vision-language model through
[Ollama](https://ollama.com/). Everything still stays on your machine — Ollama
serves the model on `localhost`.

1. **Install Ollama**

   - **Windows / macOS:** download the installer from
     [ollama.com/download](https://ollama.com/download) and run it. Ollama
     starts automatically and runs in the background.
   - **Linux:**
     ```bash
     curl -fsSL https://ollama.com/install.sh | sh
     ```

2. **Download the vision model** (one-time, requires internet)

   ```bash
   ollama pull moondream
   ```

   `moondream` is small and CPU-friendly. You can use any other vision model
   Ollama supports (e.g. `llava`) by setting `LEAF_VLM_MODEL` accordingly.

3. **Make sure the Ollama server is running**

   The desktop app keeps the server running automatically. To start it manually
   (or on Linux/servers), run:

   ```bash
   ollama serve
   ```

   By default it listens on `http://127.0.0.1:11434`, which matches the app's
   `OLLAMA_HOST` default. Verify it responds:

   ```bash
   ollama list          # shows installed models (moondream should appear)
   ```

4. **Run the Leaf Detection app** (`python run.py`). On startup the log prints
   whether the local VLM was detected:

   ```
   local VLM       : READY
   ```

   If Ollama isn't running, the app still works and falls back to the OpenCV
   colour heuristic for the health status.

> **Tip:** Start Ollama **before** launching the app so the VLM is detected on
> boot. If you start it afterwards, refresh the page and analyze again.

## Configuration

Configuration is managed in [`app/backend/config.py`](app/backend/config.py). Key settings can be overridden via environment variables:

| Environment Variable | Default | Description |
|---|---|---|
| `LEAF_EMBEDDING_MODEL` | `convnext_small` | Backbone model (`mobilenet_v3_large`, `convnext_tiny`, `convnext_small`, `efficientnet_v2_s`) |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL for VLM health analysis |
| `LEAF_VLM_MODEL` | `moondream` | Vision-language model name |
| `LEAF_SESSION_TTL` | `604800` | Login session lifetime in seconds (default 7 days) |
| `LEAF_PBKDF2_ITERATIONS` | `200000` | PBKDF2 iterations for password hashing |
| `LEAF_FACE_MATCH_THRESHOLD` | `0.86` | Cosine-similarity cutoff for accepting a face match (0–1) |
| `LEAF_FACE_MIN_SIZE` | `80` | Smallest detectable face, in pixels |

> **Important:** Changing the embedding model requires re-running setup and rebuilding the index.

## Accounts & Face Login

The app requires an account. On first launch you'll be taken to the sign-in
page (`login.html`); create an account with your name, email, and a password
(minimum 8 characters). You can optionally enable **face login** during
sign-up, or sign in later with **email + password**.

How it works, fully offline:

- **Passwords** are stored as salted **PBKDF2-HMAC-SHA256** hashes in a local
  SQLite database (`app/artifacts/users.db`, git-ignored). Nothing leaves your
  machine.
- **Face login** detects your face with OpenCV's bundled Haar cascade, crops
  it, and embeds it with the **same ConvNeXt backbone** used for leaves. Login
  compares the captured face against enrolled embeddings by cosine similarity.

> **Security note:** Face embeddings are produced by a general-purpose ImageNet
> backbone, not a dedicated face-recognition network, and there is **no liveness
> detection** — a printed photo could fool it. Treat face login as a
> convenience, not a hardened security control. Email + password is the
> primary, reliable credential. If face login rejects you too often (or accepts
> too easily), tune `LEAF_FACE_MATCH_THRESHOLD`. Because the face embedder is
> the configured `LEAF_EMBEDDING_MODEL`, changing that model invalidates any
> previously enrolled faces — re-enroll after switching backbones.

To reset all accounts, stop the app and delete `app/artifacts/users.db`.
