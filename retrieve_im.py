from PIL import Image
import io
import msgpack
import torch
from train_model_scratch import CLIPWithProjection
import open_clip
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("HEY")

# Recreate the model architecture exactly the same as during training
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)


embed_dim = model.text_projection.shape[1]
proj_dim = 256  # must match your training value

fine_tuned_model = CLIPWithProjection(model, proj_dim)
fine_tuned_model.load_state_dict(torch.load("model.pth", map_location=device))

fine_tuned_model = fine_tuned_model.to(device)
fine_tuned_model.eval()
#with torch.no_grad():
df = pd.read_csv("newsimages_25_v1.1/subset.csv", header=None)
headlines = df[2].tolist()
image_ids = df[4].tolist()
gt_images=[]
for im in image_ids:
    img_address="newsimages_25_v1.1/newsimages/"+im+".jpg"
    img = Image.open(img_address)
    gt_images.append(img)


tokenizer = open_clip.get_tokenizer('ViT-B-32')
count=0
max_count=5000

directory = os.fsencode("output_dir")
embeddings = []
ret_list=[]
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".msgpack"):
        fullname="output_dir"+"/"+filename
        with open(fullname, "rb") as f:
                unpacker = msgpack.Unpacker(f, raw=False)  # raw=False = decode strings properly
                for i, sample in enumerate(unpacker):
                    img = Image.open(io.BytesIO(sample["image"]))
                    ret_list.append(img)
                    img_tensor = preprocess(img).unsqueeze(0).to(device)  # [1, 3, H, W]
                    # Forward pass through CLIP
                    with torch.no_grad():
                        img_feat, _, _,_,_,_ = fine_tuned_model(img_tensor, tokenizer("lalaaaa"))  
                        feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # normalize

                    embeddings.append(feat.cpu())
                    count+=1
                    if count==max_count:
                        break
                if count==max_count:
                    break


# 3. Stack into one big embedding matrix
embedding_matrix = torch.cat(embeddings, dim=0)
def retrieve_image(query_text):
    pred_sec=""
    with torch.no_grad():
        # Forward pass: single dummy image batch just to get text embedding
        dummy_image = torch.zeros(1, 3, 224, 224).to(device)  # shape doesn't matter here
        _, text_feat, _,logits,_,_ = fine_tuned_model(dummy_image, tokenizer([query_text]).to(device))
        sec=logits.argmax(dim=-1)
        #section=categories[sec]
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    # Compute similarity
    sims = (text_feat.cpu() @ embedding_matrix.T).squeeze(0)  # shape: [N]
    best_idx = sims.argmax().item()
    return pred_sec,ret_list[best_idx]

def retrieve_image_original(query_text):
    with torch.no_grad():
        # Forward pass: single dummy image batch just to get text embedding
        dummy_image = torch.zeros(1, 3, 224, 224).to(device)  # shape doesn't matter here
        _, text_feat, _ = model(dummy_image, tokenizer([query_text]).to(device))
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    # Compute similarity
    sims = (text_feat.cpu() @ embedding_matrix.T).squeeze(0)  # shape: [N]
    best_idx = sims.argmax().item()
    return ret_list[best_idx]

for i in range(len(headlines)):
    headline = headlines[i]
    gt_image = gt_images[i]
    pred_sec, retrieved_image = retrieve_image(headline)

    # Show ground truth and retrieved images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(gt_image)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(retrieved_image)
    axes[1].set_title(f"Retrieved, in section:"+pred_sec)  # truncate if too long
    axes[1].axis("off")

    plt.suptitle(f"Headline: {headline}", fontsize=12, wrap=True)
    plt.tight_layout()
    plt.show()