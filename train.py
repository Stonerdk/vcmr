import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Dataset
import os
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
from torch.amp import autocast, GradScaler
from info_nce import InfoNCE
import random
from transformers import BertModel

os.chdir("/root/dev/vcmr")

num_epochs = 1000
learning_rate = 1e-4
batch_size = 256
decay = 1e-6
contrastive = True
negative_aug = 3
siamese = False
model_name = "TRANSFORMER"
temperature = 0.2 # 0.2
steplr_gamma = 0.5
steplr_step = 10
z_size = 256
loss_type = "infonce"
import pickle

def cosine_similarity_loss(image_embeds, music_embeds, labels):
    loss = F.cosine_embedding_loss(image_embeds, music_embeds, labels)
    return loss

info_nce_loss = InfoNCE(temperature=temperature)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))

class RES(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, num_blocks=5):
        super(RES, self).__init__()
        self.initial = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(num_blocks)])
        self.final = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.final(x)
        x = F.normalize(x, dim=1)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim=512, output_dim=128):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        z = self.mlp(x)
        return F.normalize(z, dim=1)

class Identical(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identical, self).__init__()

    def forward(self, x):
        return x

class SampleCNNMLP(nn.Module):
    # 프로그래밍의 일관성을 위해서 파라미터 유지, 그러나 사용하지 않음
    def __init__(self, input_dim=512, input_samplecnn_dim=50, output_dim=512):
        super(SampleCNNMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_samplecnn_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x, s):
        z = self.mlp(s)
        return F.normalize(z, dim=1)

class MusicTransformer(nn.Module):
    def __init__(self, input_dim=512, input_samplecnn_dim=50, output_dim=128, transformer_hidden_dim=256, nhead=8, num_layers=4, dropout=0.1):
        super(MusicTransformer, self).__init__()
        self.project_x = nn.Linear(input_dim, transformer_hidden_dim)
        self.project_s = nn.Linear(input_samplecnn_dim, transformer_hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(2, transformer_hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(transformer_hidden_dim, output_dim)

    def forward(self, x, s):
        x_proj = self.project_x(x)  # (batch_size, transformer_hidden_dim)
        s_proj = self.project_s(s)  # (batch_size, transformer_hidden_dim)
        combined = torch.stack([x_proj, s_proj], dim=1)  # (batch_size, 2, transformer_hidden_dim)
        combined = combined + self.positional_encoding  # (batch_size, 2, transformer_hidden_dim)
        combined = combined.permute(1, 0, 2)  # (2, batch_size, transformer_hidden_dim)
        transformer_output = self.transformer_encoder(combined)  # (2, batch_size, transformer_hidden_dim)
        aggregated = transformer_output.mean(dim=0)  # (batch_size, transformer_hidden_dim)
        out = self.output_layer(aggregated)  # (batch_size, output_dim)
        out = F.normalize(out, p=2, dim=1)  # (batch_size, output_dim)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_name == "RES":
    ImageProjection = RES
    MusicProjection = RES
elif model_name == "MLP":
    ImageProjection = MLP
    MusicProjection = MLP
elif model_name == "TRANSFORMER":
    ImageProjection = MLP
    MusicProjection = MusicTransformer
elif model_name == "PRETRAINED_TRANSFORMER":
    ImageProjection = MLP
    MusicProjection = PretrainedMusicTransformer
elif model_name == "SAMPLECNN_MLP":
    ImageProjection = Identical
    MusicProjection = SampleCNNMLP

if loss_type == "cossim":
    loss_fn = cosine_similarity_loss
elif loss_type == "infonce":
    loss_fn = lambda x, y, l: info_nce_loss(x, y)

if __name__ == "__main__":
    # HPARAMS
    wandb.init(
        project="image-music-recommendation",
        config={
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "decay": decay,
            "model": f"{model_name}-{'SIAM' if siamese else 'NOSIAM'}",
            "loss": "cosine_embedding_loss",  # Updated from "infonce_loss"
            "temperature": temperature,
            "scheduler": "StepLR",
            "scheduler_step_size": steplr_step,
            "scheduler_gamma": steplr_gamma,
            "negative_aug": negative_aug
        }
    )
    config = wandb.config
    wandb.run.name = f"processed-{config.model}-{num_epochs}-{learning_rate}-{batch_size}-{decay}"

    print("PRELOAD...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = './image_features_train'
    music_dir = './music_features_train'
    music_samplecnn_dir = './music_SampleCNN_train'

    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
    filenames = [os.path.splitext(f)[0] for f in files]
    image_embeddings = []
    music_embeddings = []
    music_samplecnn_embeddings = []
    for f in files:
        if os.path.exists(os.path.join(music_samplecnn_dir, f)) is False:
            continue
        image_embed = np.load(os.path.join(image_dir, f)).astype(np.float32)
        music_embed = np.load(os.path.join(music_dir, f)).astype(np.float32)
        music_samplecnn_embed = np.load(os.path.join(music_samplecnn_dir, f)).astype(np.float32)
        image_embeddings.append(image_embed)
        music_embeddings.append(music_embed)
        music_samplecnn_embeddings.append(music_samplecnn_embed)
    image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32)
    music_embeddings = torch.tensor(music_embeddings, dtype=torch.float32)
    music_samplecnn_embeddings = torch.tensor(music_samplecnn_embeddings, dtype=torch.float32)

    num_samples = len(image_embeddings)
    train_indices, val_indices = train_test_split(list(range(num_samples)), test_size=0.2, random_state=42)

    print("DATASET...")
    class PairDataset(Dataset):
        def __init__(self, image_embeddings, music_embeddings, music_samplecnn_embeddings, negative_aug, indices, positive=True):
            self.image_embeds = image_embeddings
            self.music_embeds = music_embeddings
            self.music_samplecnn_embeds = music_samplecnn_embeddings
            self.indices = indices
            self.negative_aug = negative_aug
            self.positive = positive
            self.labels = torch.ones(len(indices), dtype=torch.float32) if positive else torch.ones(len(indices), dtype=torch.float32) * -1.0

        def __len__(self):
            if self.positive:
                return len(self.indices)
            else:
                return len(self.indices) * self.negative_aug

        def __getitem__(self, idx):
            if self.positive:
                i = self.indices[idx]
                img = self.image_embeds[i]
                music = self.music_embeds[i]
                music_samplecnn = self.music_samplecnn_embeds[i]
                label = self.labels[idx]
            else:
                i = self.indices[idx // self.negative_aug]
                img = self.image_embeds[i]
                j = random.choice(self.indices)
                while abs(j - i) < 2:
                    j = random.choice(self.indices)
                music = self.music_embeds[j]
                music_samplecnn = self.music_samplecnn_embeds[j]
                label = self.labels[idx // self.negative_aug]
            return img, music, music_samplecnn, label

    if contrastive:
        train_positive_dataset = PairDataset(image_embeddings, music_embeddings, music_samplecnn_embeddings, negative_aug, train_indices, positive=True)
        train_negative_dataset = PairDataset(image_embeddings, music_embeddings, music_samplecnn_embeddings, negative_aug, train_indices, positive=False)
        train_dataset = ConcatDataset([train_positive_dataset, train_negative_dataset])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_positive_dataset = PairDataset(image_embeddings, music_embeddings, music_samplecnn_embeddings, negative_aug, val_indices, positive=True)
        val_negative_dataset = PairDataset(image_embeddings, music_embeddings, music_samplecnn_embeddings, negative_aug, val_indices, positive=False)
        val_dataset = ConcatDataset([val_positive_dataset, val_negative_dataset])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    else:
        train_positive_dataset = PairDataset(image_embeddings, music_embeddings, music_samplecnn_embeddings, negative_aug, train_indices, positive=True)
        train_loader = DataLoader(train_positive_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_positive_dataset = PairDataset(image_embeddings, music_embeddings, music_samplecnn_embeddings, negative_aug, val_indices, positive=True)
        val_loader = DataLoader(val_positive_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print("SETTING MODEL...")

    if siamese:
        shared_proj = ImageProjection(input_dim=512, output_dim=z_size).to(device)
        image_proj = shared_proj
        music_proj = shared_proj
    else:
        image_proj = ImageProjection(input_dim=512, output_dim=z_size).to(device)
        music_proj = MusicProjection(input_dim=512, output_dim=z_size).to(device)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    image_proj.apply(weights_init)
    music_proj.apply(weights_init)
    params = list(image_proj.parameters()) + list(music_proj.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=decay)

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steplr_step, gamma=steplr_gamma)

    wandb.watch(image_proj, log="all")
    wandb.watch(music_proj, log="all")

    print("TRAINING...")
    for epoch in range(num_epochs):
        image_proj.train()
        music_proj.train()
        train_loss = 0.0
        for image_embed, music_embed, music_samplecnn_embed, label in train_loader:
            image_embed = image_embed.to(device, non_blocking=True)
            music_embed = music_embed.to(device, non_blocking=True)
            music_samplecnn_embed = music_samplecnn_embed.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast("cuda"):
                projected_image = image_proj(image_embed)
                if model_name == "TRANSFORMER" or model_name == "SAMPLECNN_MLP":
                    projected_music = music_proj(music_embed, music_samplecnn_embed)
                else:
                    projected_music = music_proj(music_embed)
                loss = loss_fn(projected_image, projected_music, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        image_proj.eval()
        music_proj.eval()
        val_loss = 0.0
        val_cossimloss = 0.0
        with torch.no_grad():
            for image_embed, music_embed, music_samplecnn_embed, label in val_loader:
                image_embed = image_embed.to(device, non_blocking=True)
                music_embed = music_embed.to(device, non_blocking=True)
                music_samplecnn_embed = music_samplecnn_embed.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                with autocast("cuda"):
                    projected_image = image_proj(image_embed)
                    if model_name == "TRANSFORMER" or model_name == "SAMPLECNN_MLP":
                        projected_music = music_proj(music_embed, music_samplecnn_embed)
                    else:
                        projected_music = music_proj(music_embed)
                    loss = loss_fn(projected_image, projected_music, label)
                    cossim_loss = cosine_similarity_loss(projected_image, projected_music, label)
                val_loss += loss.item()
                val_cossimloss += cossim_loss.mean().item()
        avg_val_loss = val_loss / len(val_loader)
        avg_val_cossimloss = val_cossimloss / len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": scheduler.get_last_lr()[0],
        })
        if (epoch + 1) % 100 == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                "epoch": epoch + 1,
                "image_proj_state_dict": image_proj.state_dict(),
                "music_proj_state_dict": music_proj.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_val_loss,
            }
            # Save the model class along with the checkpoint
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}_samplecnn.pth"
            torch.save(checkpoint, checkpoint_path)
            # wandb.save(checkpoint_path)

    wandb.finish()