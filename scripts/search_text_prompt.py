import torch
import clip

from src.indexer import load_faiss_index, get_image_id_mapping
from src.config import IMAGE_FOLDER
from PIL import Image
import os

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def embed_text(prompt):
    tokens = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        vec = model.encode_text(tokens)
        vec /= vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy()

def show_results(indices, id_to_filename):
    print("\nTop Matches:")
    for i in indices[0]:
        filename = id_to_filename.get(i, "<unknown>")
        print(f"{filename}")
        
        try:
            img = Image.open(os.path.join(IMAGE_FOLDER, filename))
            img.show()
        except Exception as e:
            print(f"Could not open image: {e}")
        

if __name__ == "__main__":
    while (True):
        print("Prompt2PictureAgent Search")
        prompt = input("Enter your prompt: ")

        text_vec = embed_text(prompt)

        index = load_faiss_index()
        id_to_filename = get_image_id_mapping()

        # Search (top 5)
        D, I = index.search(text_vec, k=5)

        show_results(I, id_to_filename)
