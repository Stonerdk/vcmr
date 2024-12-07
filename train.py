import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Dataset
import os
import numpy as np
from sklearn.model_selection import train_test_split
import wandb

os.chdir("/root/dev/vcmr")

def cosine_similarity_loss(image_embeds, music_embeds, labels):
    loss = F.cosine_embedding_loss(image_embeds, music_embeds, labels)
    return loss

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        z = self.mlp(x)
        return F.normalize(z)

if __name__ == "__main__":
    # HPARAMS
    num_epochs = 100
    learning_rate = 1e-3
    batch_size = 256
    decay = 0.01
    constrastive = False
    siamese = True

    wandb.init(
        project="image-music-recommendation",
        config={
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "decay": decay,
            "model": f"MLP-{'SIAM' if siamese else 'NOSIAM'}",
            "loss": "cosine_embedding_loss"
        }
    )
    config = wandb.config
    wandb.run.name = f"{config.model}-{num_epochs}-{learning_rate}-{batch_size}-{decay}"

    print("PRELOAD...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = './image_features_train'
    music_dir = './music_features_train'

    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
    filenames = [os.path.splitext(f)[0] for f in files]
    image_embeddings = []
    music_embeddings = []
    for f in files:
        image_embed = np.load(os.path.join(image_dir, f)).astype(np.float32)
        music_embed = np.load(os.path.join(music_dir, f)).astype(np.float32)
        image_embeddings.append(image_embed)
        music_embeddings.append(music_embed)
    image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32)
    music_embeddings = torch.tensor(music_embeddings, dtype=torch.float32)

    num_samples = len(filenames)
    num_image_embeddings = image_embeddings.size(1)
    num_music_embeddings = music_embeddings.size(1)
    print(num_samples, num_image_embeddings, num_music_embeddings)

    train_indices, val_indices = train_test_split(list(range(num_samples)), test_size=0.2, random_state=42)

    print("DATASET...")
    class PairDataset(Dataset):
        def __init__(self, image_embeds, music_embeds, indices, positive=True):
            self.image_embeds = image_embeds
            self.music_embeds = music_embeds
            self.indices = indices
            self.positive = positive
            self.labels = torch.ones(len(indices), dtype=torch.float32) if positive else torch.ones(len(indices), dtype=torch.float32) * -1.0

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            i = self.indices[idx]
            img = self.image_embeds[i]
            if self.positive:
                music = self.music_embeds[i]
                label = self.labels[idx]
            else:
                j = (i + 4) % self.image_embeds.size(0)
                music = self.music_embeds[j]
                label = self.labels[idx]
            return img, music, label

    if constrastive:
        train_positive_dataset = PairDataset(image_embeddings, music_embeddings, train_indices, positive=True)
        train_negative_dataset = PairDataset(image_embeddings, music_embeddings, train_indices, positive=False)
        val_positive_dataset = PairDataset(image_embeddings, music_embeddings, val_indices, positive=True)
        val_negative_dataset = PairDataset(image_embeddings, music_embeddings, val_indices, positive=False)
        train_dataset = ConcatDataset([train_positive_dataset, train_negative_dataset])
        val_dataset = ConcatDataset([val_positive_dataset, val_negative_dataset])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        train_positive_dataset = PairDataset(image_embeddings, music_embeddings, train_indices, positive=True)
        val_positive_dataset = PairDataset(image_embeddings, music_embeddings, val_indices, positive=True)
        train_loader = DataLoader(train_positive_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_positive_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("SETTING MODEL...")


    shared_proj = ProjectionHead(input_dim=512, output_dim=128).to(device)
    music_proj = shared_proj
    image_proj = shared_proj

    optimizer = torch.optim.Adam(
        list(image_proj.parameters()) + list(music_proj.parameters()), lr=learning_rate, weight_decay=decay
    )

    from torch.amp import autocast, GradScaler

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)


    wandb.watch(image_proj, log="all")
    wandb.watch(music_proj, log="all")

    print("TRAINING...")
    for epoch in range(num_epochs):
        image_proj.train()
        music_proj.train()
        train_loss = 0.0
        for image_embed, music_embed, label in train_loader:
            image_embed = image_embed.to(device, non_blocking=True)
            music_embed = music_embed.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast("cuda"):
                projected_image = image_proj(image_embed)
                projected_music = music_proj(music_embed)
                loss = cosine_similarity_loss(projected_image, projected_music, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        image_proj.eval()
        music_proj.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image_embed, music_embed, label in val_loader:
                image_embed = image_embed.to(device, non_blocking=True)
                music_embed = music_embed.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                with autocast("cuda"):
                    projected_image = image_proj(image_embed)
                    projected_music = music_proj(music_embed)
                    loss = cosine_similarity_loss(projected_image, projected_music, label)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        if (epoch + 1) % 20 == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                "epoch": epoch + 1,
                "image_proj_state_dict": image_proj.state_dict(),
                "music_proj_state_dict": music_proj.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_val_loss,
            }
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, checkpoint_path)
            wandb.save(checkpoint_path)

    wandb.finish()