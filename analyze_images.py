import pickle
import faiss
import pandas as pd
import torch
from model import longclip
from train_model import CLIPWithProjection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FAISS index + keys
index = faiss.read_index("/scratch-shared/bbakker/image_index_real.faiss")
with open("keys.pkl", "rb") as f:
    all_keys = pickle.load(f)

model, preprocess = longclip.load("/home/bbakker/checkpoints/longclip-B.pt", device=device)
embed_dim = model.text_projection.shape[1]
proj_dim = 512  # must match your training value

fine_tuned_model = CLIPWithProjection(model, proj_dim)
fine_tuned_model.load_state_dict(torch.load("text_model.pth", map_location=device))
fine_tuned_model = fine_tuned_model.to(device).eval()

assert index.ntotal == len(all_keys), "Mismatch between FAISS index and keys"

# Load headlines
df = pd.read_csv("/scratch-shared/bbakker/newsimages_25_v1.1/newsarticles.csv", header=None)
headlines = df[2].tolist()
#article = df[6].tolist()
#combined = [f"{h}: {a}" for h, a in zip(headlines, articles)]

# Batch encode all text
batch_size = 128
all_embeddings = []

with torch.no_grad():
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i+batch_size]
        tokens = longclip.tokenize(batch, truncate=True).to(device)
        text_emb = model.encode_text(tokens)
        #dummy_image = torch.zeros(1, 3, 224, 224).to(device) 
        #_, text_emb, section = fine_tuned_model(dummy_image, longclip.tokenize([h],truncate=True).to(device))
        #sections.append(section)
        text_emb = torch.nn.functional.normalize(text_emb.float(), dim=-1)
        all_embeddings.append(text_emb.cpu())

all_embeddings = torch.cat(all_embeddings, dim=0).numpy().astype("float32")

# FAISS search in one go
_, ids = index.search(all_embeddings, k=1)

# Map indices to keys
strings = [all_keys[idx[0]] for idx in ids]

# Save results
with open("/scratch-shared/bbakker/image_strings.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(strings))

#with open("/scratch-shared/bbakker/subset_sections.txt", "w", encoding="utf-8") as f:
    #for s in sections:
        #f.write(s[0] + "\n")
