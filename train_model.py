import json
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import open_clip
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import shutil
from model import longclip


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model, preprocess = longclip.load("/home/bbakker/im_retrieval/checkpoints/longclip-B.pt", device=device)
model = model.float()

for param in model.parameters():
    param.requires_grad = False

embed_dim = model.text_projection.shape[1] 

class LearnableAlignmentModule(nn.Module):
    def __init__(self, num_categories, embed_dim):
        super().__init__()
        self.category_embeddings = nn.Parameter(torch.randn(num_categories, embed_dim))

        self.category_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_categories)
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, text_features, labels=None):

        logits = self.category_predictor(text_features)   # [B, C]
        w_x = F.softmax(logits, dim=-1)                   # [B, C]

        # Weighted category embedding
        cat_embedding = w_x @ self.category_embeddings     # [B, d]

        # Combine with text features
        h_lam = self.alpha * text_features + (1 - self.alpha) * cat_embedding
        if labels!= None:
            cat_loss = F.cross_entropy(logits, labels)
            return h_lam, logits, cat_loss
        else:
            return h_lam, logits
    

class CLIPWithProjection(nn.Module):
    def __init__(self, clip_model, proj_dim=256, num_categories=24, id2section=None):
        super().__init__()
        self.clip_model = clip_model


        self.text_proj = nn.Linear(embed_dim, proj_dim)

        self.lam = LearnableAlignmentModule(num_categories, proj_dim) if num_categories else None
        self.id2section = id2section

    def forward(self, images, text_tokens, labels=None):

        with torch.no_grad():
            img_features = self.clip_model.encode_image(images)

        txt_features = self.clip_model.encode_text(text_tokens)

        img_emb = img_features / img_features.norm(dim=-1, keepdim=True)

        txt_emb = self.text_proj(txt_features)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        if labels is not None:
            txt_emb, logits, cat_loss = self.lam(txt_emb, labels)
        else:
            txt_emb, logits = self.lam(txt_emb)

        pred_ids = logits.argmax(dim=-1).tolist()
        section_str = [self.id2section[i] for i in pred_ids]

        if labels is not None:
            return img_emb, txt_emb, txt_features.detach(), cat_loss, section_str
        else:
            return img_emb, txt_emb, section_str


def contrastive_loss_text_only(image_embeddings, text_embeddings, temperature=0.07):
    logits = image_embeddings @ text_embeddings.t() / temperature
    labels = torch.arange(len(logits), device=image_embeddings.device)

    return F.cross_entropy(logits.t(), labels)


def kl_divergence_loss(student_logits, teacher_logits, temperature):
    s_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s_probs, t_probs, reduction='batchmean') * (temperature ** 2)

with open(r"/scratch/bbakker/news/nytimes_train.json") as f:
    data = json.load(f)
sections = sorted({item['section'] for item in data})
section2id = {sec: i for i, sec in enumerate(sections)}
id2section = {i: sec for sec, i in section2id.items()}


proj_dim = 512
fine_tuned_model = CLIPWithProjection(model, proj_dim,id2section=id2section).to(device).float()

optimizer = optim.Adam(
    list(fine_tuned_model.text_proj.parameters()) +
    list(fine_tuned_model.lam.parameters()),
    lr=1e-4
)

temperature = 0.07
distill_temp = 0.5

texts=[]
ims=[]


class ImageTextDataset(Dataset):
    def __init__(self, data, preprocess, section2id, max_count=None):
        self.samples = []
        for item in data:
            im_id = item['image_id']
            headline = item['headline']
            article = item['article']
            full_text=headline+": "+article
            section = section2id[item['section']]  # map to int
            self.samples.append((f"/scratch/bbakker/imgs/imgs/{im_id}.jpg", full_text, section))
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, full_text, section = self.samples[idx]
        image = self.preprocess(Image.open(img_path))
        return image, full_text, section 
    

if __name__ == "__main__":
    num_categories = len(section2id)
    dataset = ImageTextDataset(data, preprocess, section2id, max_count=None)

    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(6):
        for images, texts, labels in dataloader:
            images = images.to(device)
            text_tokens = longclip.tokenize(texts, truncate=True).to(device)
            labels = labels.to(device)

            img_emb, txt_emb, txt_teacher, cat_loss, _ = fine_tuned_model(images, text_tokens, labels)

            loss_contrastive = contrastive_loss_text_only(img_emb, txt_emb, temperature)

            student_logits = txt_emb @ img_emb.t()    
            teacher_logits = txt_teacher @ img_emb.t()
            loss_kld = kl_divergence_loss(student_logits, teacher_logits, distill_temp)

            loss = loss_contrastive + 0.3 * loss_kld + 0.3 * cat_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    torch.save(fine_tuned_model.state_dict(), "/scratch/bbakker/output_dir/sentence_model.pth")
