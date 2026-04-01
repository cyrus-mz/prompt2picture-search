# Prompt2PictureSearch

Prompt2PictureSearch is a local semantic photo search application for personal image collections. It lets users ingest image folders from their machine, index them with CLIP and FAISS, search with natural language, inspect library health, and rebuild the search index when files change.

The current version focuses on a practical local workflow:
- folder-based image ingestion from user-provided directories
- semantic search over indexed personal images
- library scan and index rebuild utilities
- a Streamlit UI for search and library management

## Highlights

- **Local-first workflow**: images remain on the user's machine
- **Natural language photo search** powered by CLIP text-image embeddings
- **Incremental image ingestion** for newly indexed folders
- **Fast vector retrieval** with FAISS
- **Library health checks** to identify missing or invalid files
- **Index rebuild support** to resynchronize the searchable collection after file changes
- **Modular architecture** designed for future extension into captioning, image-to-image search, and agent/tool workflows

## Current Features

### Semantic search
Users can search their indexed image collection with prompts such as:
- `a dog running in the grass`
- `children playing outdoors`
- `a person riding a horse`

Search results are shown in a gallery view in the Streamlit UI.

### Image ingestion
The app can ingest images from a local folder path, compute CLIP embeddings, and append them to the FAISS index.

### Library scan
The app can scan indexed records and report whether files are:
- available
- missing
- invalid

### Index rebuild
The app can rebuild the FAISS index from the current database records, re-embedding valid images and excluding unavailable ones from search.

## Architecture

The project is structured around a service-oriented backend.

### Core layers

- **Model layer**: CLIP loading and embedding utilities
- **Storage layer**: FAISS index management and metadata-to-result mapping
- **Service layer**: search, ingestion, scan, and rebuild workflows
- **UI layer**: Streamlit application for end-user interaction

## Requirements

- Python 3.10+
- A virtual environment is strongly recommended
- Optional but helpful: CUDA-capable GPU for faster embedding and indexing

## Installation

```bash
git clone https://github.com/cyrus-mz/prompt2picture-search.git
cd prompt2picture-search

python -m venv .venv
source .venv/bin/activate
# On Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

## Local Data Setup

The repository intentionally ignores the `data/` directory. Generated metadata and index artifacts are local runtime files and are not committed to Git.

Create the data directory locally if needed:

```bash
mkdir -p data
```

The application will populate files such as:
- `data/image_db.json`
- `data/image_index.faiss`

## Running the App

Launch the Streamlit UI from the project root:

```bash
PYTHONPATH=. streamlit run src/ui/streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal, typically:

```text
http://localhost:8501
```

## Using the App

### 1. Ingest a folder
In the **Ingest** tab:
- enter a local folder path containing images
- choose whether subfolders should be scanned recursively
- choose a batch size for embedding
- run ingestion

The app will validate image files, compute hashes, avoid exact duplicates, generate embeddings, and update the FAISS index.

### 2. Search the collection
In the **Search** tab:
- enter a natural language prompt
- select the number of results to retrieve
- view matches in a gallery layout

### 3. Check library health
In the **Library** tab:
- run **Scan Library** to inspect indexed records
- review available, missing, and invalid records

### 4. Rebuild the index
Also in the **Library** tab:
- run **Rebuild Index** to regenerate the FAISS index from the current records
- valid files are re-embedded
- missing or invalid files are excluded from active search results

## CLI / Development Utilities

The repository also includes lightweight test and development scripts.

Examples:

```bash
PYTHONPATH=. python scripts/test_clip_model.py
PYTHONPATH=. python scripts/test_clip_batch.py
PYTHONPATH=. python scripts/test_search_service.py
PYTHONPATH=. python scripts/test_ingestion_service.py /absolute/path/to/photo1.jpg
PYTHONPATH=. python scripts/test_library_service.py scan
PYTHONPATH=. python scripts/test_library_service.py rebuild --batch-size 16
```

## Data Model Notes

Indexed image records are stored in `data/image_db.json`. Each record may include fields such as:
- `filename`
- `source_path`
- `embedding`
- `file_hash`
- `ingested_at`
- `last_scan_status`
- `last_indexed_at`

The FAISS index is stored separately in `data/image_index.faiss`.

## Design Notes

### Why folder-based ingestion?
The app is designed for local personal photo collections. It indexes images from user-provided folders and keeps the source paths in the metadata database so the collection can be searched without duplicating the original files.

### Why scan and rebuild?
Local photo collections can change over time. Files may be moved, renamed, or deleted outside the app. The scan and rebuild services help the application remain consistent with the current state of the user's collection.

## Current Limitations

- Captioning is not implemented yet
- Image-to-image search is not implemented yet
- The app currently relies on local filesystem paths and does not yet support browser-upload ingestion as a primary workflow
- Metadata filtering, reranking, and rich result annotations are still limited
- The current version is designed for local usage rather than multi-user deployment

## Roadmap

Planned improvements include:
- image captioning
- image-to-image similarity search
- richer result details and metadata filters
- gallery-level sorting and organization improvements
- tool-based interfaces for future agent orchestration
- optional API layer for broader integration

## Technical Stack

- Python
- PyTorch
- OpenAI CLIP
- FAISS
- Streamlit
- Pillow
- NumPy

## Development Notes

During development, the project was tested with the Flickr8k dataset as a convenient sample image collection. The application itself is not tied to Flickr8k and is intended to work with user-provided local image folders.

## Acknowledgments

- OpenAI CLIP for multimodal text-image embeddings
- FAISS for efficient vector similarity search
- Flickr8k for development-time testing