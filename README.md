# Prompt2PictureAgent

A local semantic image search MVP for personal photo collections. Prompt2PictureAgent lets users index a folder of images, generate CLIP embeddings, build a FAISS vector index, and retrieve the most relevant photos from a natural language query.

The current version is focused on the retrieval backbone:
- local image library ingestion from `data/images/`
- CLIP-based image embeddings
- FAISS-based vector search
- CLI semantic search with natural language prompts

This repository is intended as the foundation for a larger multimodal application that will later support richer capabilities such as image-to-image search, captioning, upload workflows, and tool-based/agentic orchestration.

## Features

- Local-first workflow, images stay on the user's machine
- Semantic photo search using CLIP text-image alignment
- Batch image embedding with preprocessing cache
- FAISS index generation for fast retrieval
- Modular Python structure with separate config, database, embedding, and indexing components
- Simple CLI scripts for database creation, embedding/indexing, and interactive search

## Project Structure

```text
prompt2picture-agent/
├── scripts/
│   ├── build_image_db.py
│   ├── embed_images.py
│   └── search_text_prompt.py
├── src/
│   ├── config.py
│   ├── image_db.py
│   ├── embedder.py
│   └── indexer.py
├── data/
│   ├── images/
│   ├── preprocessed/
│   ├── image_db.json
│   └── image_index.faiss
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

1. Place images in `data/images/`.
2. Build the local metadata database.
3. Preprocess and embed images using CLIP.
4. Build a FAISS index from the image embeddings.
5. Enter a natural language prompt and retrieve the top matching images.

## Requirements

- Python 3.10+
- Optional but recommended: CUDA-capable GPU for faster embedding and search

## Installation

```bash
# Clone the repository
git clone https://github.com/kurosh90/prompt2picture-agent.git
cd prompt2picture-agent

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
# On Windows PowerShell:
# .venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Setup

This repository ignores the `data/` directory on purpose. The application is designed to work with user-provided images, so datasets, cached tensors, index files, and generated metadata are not stored in Git.

Create the following directories locally if they do not already exist:

```bash
mkdir -p data/images
mkdir -p data/preprocessed
```

Then place your image files inside `data/images/`.

### Example dataset

During development, the project was tested with the Flickr8k image dataset. You do not need Flickr8k specifically to run the application, any local image collection with supported formats (`.jpg`, `.jpeg`, `.png`) can be used.

## Usage

### 1. Build the image database

```bash
PYTHONPATH=. python scripts/build_image_db.py
```

This scans `data/images/` and creates `data/image_db.json`.

### 2. Embed images and build the FAISS index

```bash
PYTHONPATH=. python scripts/embed_images.py
```

This will:
- preprocess and cache image tensors in `data/preprocessed/`
- generate CLIP embeddings
- store embeddings in `data/image_db.json`
- build `data/image_index.faiss`

### 3. Run interactive search

```bash
PYTHONPATH=. python scripts/search_text_prompt.py
```

Then enter prompts such as:
- `a dog running on the beach`
- `children playing outdoors`
- `a person riding a horse`

The script prints the top matching filenames and attempts to open the matching images locally.


## Current Limitations

- Search is currently CLI-only
- The app does not yet support direct user uploads through a UI
- Captioning and image-to-image search are not implemented yet
- Metadata filtering and result reranking are not yet available
- Index updates are driven through scripts rather than an application service layer

## Roadmap

Planned improvements include:
- user upload workflow
- incremental indexing for newly added photos
- image-to-image similarity search
- caption generation for result enrichment
- web UI with gallery display
- modular service/tool layer for later agent integration

## Technical Stack

- Python
- PyTorch
- OpenAI CLIP
- FAISS

## Reproducibility Notes

The project stores generated artifacts locally and outside version control:
- `data/image_db.json`
- `data/image_index.faiss`
- `data/preprocessed/`

This keeps the repository lightweight and allows users to index their own photo collections.

## Acknowledgments

- OpenAI CLIP for text-image embedding
- FAISS for similarity search
- Flickr8k for development-time testing