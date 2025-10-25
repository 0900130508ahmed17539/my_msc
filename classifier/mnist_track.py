"""
Complete U-Net Flow Matching with IMPROVED Classifier Guidance for MNIST
Fixed memory issues and feature extraction bottlenecks
TRAINING: 50 epochs | EVALUATION: 10k samples
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import math
import os
import gc
import time
from typing import Tuple
from dataclasses import dataclass
from collections import namedtuple
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    import sys
    print("Installing tqdm...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_FOLDER = 'mnist_flow_50ep'
NUM_EPOCHS_FLOW = 50
NUM_EPOCHS_CLASSIFIER = 50
NUM_SAMPLES_EVAL = 10000  # Can reduce to 5000 if memory issues persist

# ============================================================================
# DEVICE SETUP
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ============================================================================
# SIMPLIFIED FID COMPUTATION FOR MNIST
# ============================================================================
class SimplifiedPrecisionRecall:
    """Simplified P&R computation with VGG16"""
    
    def __init__(self, k=3, device="cuda"):
        self.k = k
        self.device = device
        
        # Load VGG16
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.vgg16.classifier = self.vgg16.classifier[:4]
        self.vgg16.eval()
        
        for p in self.vgg16.parameters():
            p.requires_grad = False
        
        self.vgg16 = self.vgg16.to(device)
        
        if torch.cuda.is_available():
            self.vgg16 = self.vgg16.half()
    
    @torch.no_grad()
    def extract_features_fast(self, dataloader, max_samples=None):
        """Extract features with optimizations"""
        features_list = []
        samples_processed = 0
        
        for batch in tqdm(dataloader, desc="Extracting features"):
            if max_samples and samples_processed >= max_samples:
                break
            
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            if max_samples:
                remaining = max_samples - samples_processed
                batch = batch[:min(len(batch), remaining)]
            
            if batch.shape[-1] != 224:
                batch = F.interpolate(batch, size=(224, 224), 
                                    mode='bilinear', align_corners=False)
            
            batch = batch.to(self.device)
            
            if torch.cuda.is_available():
                batch = batch.half()
            
            features = self.vgg16(batch)
            features_list.append(features.float().cpu().numpy())
            
            samples_processed += len(batch)
        
        return np.concatenate(features_list, axis=0)
    
    def compute_distances_efficient(self, X, Y):
        """Compute distances using vectorized operations"""
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
        
        distances = 1 - np.dot(X, Y.T)
        
        return distances
    
    def compute_radii_fast(self, features):
        """Compute k-NN radii efficiently"""
        n = len(features)
        radii = np.zeros(n, dtype=np.float32)
        
        chunk_size = min(1000, n)
        
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            
            chunk_dists = self.compute_distances_efficient(
                features[i:end_i], features
            )
            
            for j in range(end_i - i):
                kth_dists = np.partition(chunk_dists[j], self.k + 1)[:self.k + 1]
                radii[i + j] = kth_dists[-1]
        
        return radii
    
    def compute_metrics_fast(self, ref_features, ref_radii, gen_features):
        """Compute precision and recall efficiently"""
        n_ref = len(ref_features)
        n_gen = len(gen_features)
        
        if n_ref * n_gen < 1e8:
            distances = self.compute_distances_efficient(ref_features, gen_features)
            precision = np.mean(np.any(distances < ref_radii[:, np.newaxis], axis=0))
            gen_radii = self.compute_radii_fast(gen_features)
            distances_T = distances.T
            recall = np.mean(np.any(distances_T < gen_radii[:, np.newaxis], axis=0))
        else:
            precision = self._compute_precision_batched(ref_features, ref_radii, gen_features)
            recall = self._compute_recall_batched(ref_features, gen_features)
        
        return precision, recall
    
    def _compute_precision_batched(self, ref_features, ref_radii, gen_features):
        n_gen = len(gen_features)
        batch_size = 100
        precision_count = 0
        
        for i in range(0, n_gen, batch_size):
            batch = gen_features[i:min(i + batch_size, n_gen)]
            dists = self.compute_distances_efficient(ref_features, batch)
            precision_count += np.sum(np.any(dists < ref_radii[:, np.newaxis], axis=0))
        
        return precision_count / n_gen
    
    def _compute_recall_batched(self, ref_features, gen_features):
        gen_radii = self.compute_radii_fast(gen_features)
        n_ref = len(ref_features)
        batch_size = 100
        recall_count = 0
        
        for i in range(0, n_ref, batch_size):
            batch = ref_features[i:min(i + batch_size, n_ref)]
            dists = self.compute_distances_efficient(gen_features, batch)
            recall_count += np.sum(np.any(dists < gen_radii[:, np.newaxis], axis=0))
        
        return recall_count / n_ref
    
    def compute_simple_pr(self, real_features, fake_features):
        """Compute simplified precision and recall using existing features"""
        # Convert torch tensors to numpy if needed
        if torch.is_tensor(real_features):
            real_features = real_features.cpu().numpy()
        if torch.is_tensor(fake_features):
            fake_features = fake_features.cpu().numpy()
        
        # Compute radii for real features
        ref_radii = self.compute_radii_fast(real_features)
        
        # Compute precision and recall
        precision, recall = self.compute_metrics_fast(real_features, ref_radii, fake_features)
        
        return float(precision), float(recall)

# ============================================================================
# FID COMPUTATION
# ============================================================================

class SimplifiedFIDExtractor:
    """Simplified feature extractor that uses InceptionV3 more efficiently"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.fc = nn.Identity()
        self.model.to(device)
        self.model.eval()
        
        # Disable gradients
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    @torch.no_grad()
    def extract_features_batch(self, images, batch_size=8):
        """Extract features with aggressive memory management"""
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        features = []
        num_batches = (len(images) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(images), batch_size), 
                     desc="Extracting Inception features", 
                     total=num_batches):
            batch = images[i:i+batch_size]
            
            # Convert grayscale to RGB
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            
            # Normalize to [0, 1] if needed
            if batch.min() < 0:
                batch = (batch + 1) / 2
            
            # Resize to 299x299
            if batch.shape[-1] != 299:
                batch = F.interpolate(batch, size=(299, 299), 
                                    mode='bilinear', align_corners=False)
            
            batch = batch.to(self.device)
            batch = self.normalize(batch)
            
            # Extract features
            feat = self.model(batch)
            features.append(feat.cpu())
            
            # Aggressive cleanup
            del batch, feat
            if i % 3 == 0:
                torch.cuda.empty_cache()
        
        return torch.cat(features, dim=0)

def calculate_fid_simple(real_features, fake_features):
    """Calculate FID with numerical stability"""
    mu1 = real_features.mean(0).numpy()
    mu2 = fake_features.mean(0).numpy()
    
    sigma1 = np.cov(real_features.numpy(), rowvar=False)
    sigma2 = np.cov(fake_features.numpy(), rowvar=False)
    
    diff = mu1 - mu2
    
    # Numerical stability
    eps = 1e-6
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])
    
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid)

# ============================================================================
# FLOW MATCHING COMPONENTS
# ============================================================================

class CondOTFlowMatching:
    """Conditional Optimal Transport Flow Matching"""
    
    def __init__(self, sigma_min=0.001):
        self.sigma_min = sigma_min
    
    def sample_xt(self, x0, x1, t):
        """Linear interpolation between noise x0 and data x1"""
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x0 + t * x1
    
    def compute_ut(self, x0, x1):
        """Compute the conditional flow velocity: u_t = x1 - x0"""
        return x1 - x0

# ============================================================================
# U-NET COMPONENTS (Simplified for MNIST)
# ============================================================================

def zero_module(module):
    """Zero out the parameters of a module."""
    for p in module.parameters():
        p.detach().zero_()
    return module

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):
    """GroupNorm with 32 groups."""
    return GroupNorm32(min(32, channels), channels)

def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution layer."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):
    """Create a linear module."""
    return nn.Linear(*args, **kwargs)

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ResBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = nn.Upsample(scale_factor=2, mode="nearest")
            self.x_upd = nn.Upsample(scale_factor=2, mode="nearest")
        elif down:
            self.h_upd = nn.AvgPool2d(2)
            self.x_upd = nn.AvgPool2d(2)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            h = self.in_layers[:-1](x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = self.in_layers[-1](h)
        else:
            h = self.in_layers(x)
            
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
            
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
            
        return self.skip_connection(x) + h

class QKVAttention(nn.Module):
    """QKV attention mechanism."""
    
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

class AttentionBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels, num_heads=1, num_head_channels=-1):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
            
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class UNetModelMNIST(nn.Module):
    """U-Net model for MNIST flow matching."""
    
    def __init__(
        self,
        image_size=28,
        in_channels=1,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(14,),
        dropout=0.0,
        channel_mult=(1, 2, 2),
        dims=2,
        num_heads=4,
        num_head_channels=32,
        use_scale_shift_norm=True,
        resblock_updown=True,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            nn.Sequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        input_block_chans = [ch]
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else nn.AvgPool2d(2)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = nn.Sequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else nn.Upsample(scale_factor=2, mode="nearest")
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, t):
        """Apply the model to an input batch."""
        emb = self.time_embed(timestep_embedding(t, self.model_channels))
        
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    else:
                        h = layer(h)
            hs.append(h)

        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)

        return self.out(h)

# ============================================================================
# SIMPLIFIED CLASSIFIER FOR MNIST
# ============================================================================

class SimplifiedMNISTClassifier(nn.Module):
    """Simplified but effective classifier for MNIST"""
    
    def __init__(
        self,
        num_classes=10,
        time_emb_dim=128,
        base_channels=64,
        dropout=0.1
    ):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            linear(time_emb_dim, time_emb_dim * 4),
            SiLU(),
            linear(time_emb_dim * 4, time_emb_dim),
        )
        
        # Simple CNN backbone
        self.conv1 = nn.Conv2d(1, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head
        self.classifier = nn.Sequential(
            linear(base_channels * 8 + time_emb_dim, 256),
            SiLU(),
            nn.Dropout(dropout),
            linear(256, num_classes)
        )
        
        self.time_emb_dim = time_emb_dim
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(timestep_embedding(t, self.time_emb_dim))
        
        # CNN forward
        h = F.silu(self.conv1(x))
        h = F.silu(self.conv2(h))
        h = F.silu(self.conv3(h))
        h = F.silu(self.conv4(h))
        
        # Pool and combine with time
        h = self.pool(h).flatten(1)
        h = torch.cat([h, t_emb], dim=1)
        
        return self.classifier(h)

# ============================================================================
# EMA
# ============================================================================

class EMA(nn.Module):
    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.model = model
        self.decay = decay
        
        import copy
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        
        self.register_buffer('num_updates', torch.tensor(0))
    
    @torch.no_grad()
    def update_ema(self):
        """Update EMA parameters"""
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
    
    def forward(self, *args, **kwargs):
        """Forward pass through the main model"""
        return self.model(*args, **kwargs)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_flow_matching(
    model, 
    dataloader, 
    num_epochs=50,
    lr=2e-4,
    batch_size=128,
    physical_batch_size=32,
    use_ema=True,
    ema_decay=0.9999,
    grad_clip=1.0,
    device='cuda'
):
    """Train Flow Matching model with gradient accumulation for MNIST"""
    
    if use_ema:
        model = EMA(model, decay=ema_decay)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    scaler = amp.GradScaler()
    flow_matching = CondOTFlowMatching()
    
    accumulation_steps = batch_size // physical_batch_size
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (x1, _) in enumerate(pbar):
            x1 = x1.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            for i in range(0, x1.size(0), physical_batch_size):
                mini_x1 = x1[i:i+physical_batch_size]
                mini_batch_size = mini_x1.size(0)
                
                x0 = torch.randn_like(mini_x1)
                t = torch.rand(mini_batch_size, device=device)
                
                xt = flow_matching.sample_xt(x0, mini_x1, t)
                ut = flow_matching.compute_ut(x0, mini_x1)
                
                with amp.autocast():
                    pred = model(xt, t)
                    loss = F.mse_loss(pred, ut) / accumulation_steps
                
                scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            
            if use_ema:
                model.update_ema()
            
            epoch_losses.append(loss.item() * accumulation_steps)
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
        
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}')
        
        if (epoch + 1) % 10 == 0:
            os.makedirs(f'{OUTPUT_FOLDER}/checkpoints', exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.ema_model.state_dict() if use_ema else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'{OUTPUT_FOLDER}/checkpoints/flow_epoch_{epoch+1}.pt')
    
    return model, losses

def train_flow_classifier(
    classifier,
    flow_model,
    dataloader,
    num_epochs=50,
    lr=5e-4,
    warmup_epochs=5,
    device='cuda'
):
    """Train classifier on flow-matched interpolated data for MNIST"""
    
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)
    
    # Warmup + Cosine schedule
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = amp.GradScaler()
    flow_matching = CondOTFlowMatching()
    
    classifier.train()
    losses = []
    
    print("\n" + "="*70)
    print("TRAINING CLASSIFIER")
    print("="*70)
    
    for epoch in range(num_epochs):
        epoch_losses = []
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f'Classifier Epoch {epoch+1}/{num_epochs}')
        for x1, labels in pbar:
            x1 = x1.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = x1.shape[0]
            
            t = torch.rand(batch_size, device=device)
            x0 = torch.randn_like(x1)
            x_t = flow_matching.sample_xt(x0, x1, t)
            
            optimizer.zero_grad(set_to_none=True)
            
            with amp.autocast():
                logits = classifier(x_t, t)
                loss = F.cross_entropy(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += batch_size
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'acc': f'{correct/total:.4f}'
            })
        
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        accuracy = correct / total
        losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')
        
        if (epoch + 1) % 10 == 0:
            os.makedirs(f'{OUTPUT_FOLDER}/checkpoints', exist_ok=True)
            torch.save(classifier.state_dict(), f'{OUTPUT_FOLDER}/checkpoints/classifier_epoch_{epoch+1}.pt')
    
    return classifier, losses

# ============================================================================
# CLASSIFIER PERFORMANCE EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_classifier_performance(classifier, flow_model, test_loader, device='cuda'):
    """
    Evaluate classifier performance on:
    1. Clean test images (t=1.0)
    2. Flow-interpolated images at various timesteps
    3. Per-class accuracy breakdown
    """
    
    classifier.eval()
    flow_matching = CondOTFlowMatching()
    
    print("\n" + "="*80)
    print("CLASSIFIER PERFORMANCE EVALUATION")
    print("="*80)
    
    # Test on clean images (t=1.0)
    print("\n1. Performance on CLEAN test images (t=1.0):")
    print("-" * 80)
    
    correct_clean = 0
    total_clean = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    for images, labels in tqdm(test_loader, desc="Evaluating on clean images"):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.shape[0]
        
        # Evaluate at t=1.0 (clean data endpoint)
        t = torch.ones(batch_size, device=device)
        
        with amp.autocast():
            logits = classifier(images, t)
        
        pred = logits.argmax(dim=-1)
        correct_clean += (pred == labels).sum().item()
        total_clean += batch_size
        
        # Per-class accuracy
        for i in range(batch_size):
            label = labels[i].item()
            class_correct[label] += (pred[i] == labels[i]).item()
            class_total[label] += 1
    
    accuracy_clean = correct_clean / total_clean
    print(f"Overall Accuracy on Clean Images: {accuracy_clean:.4f} ({correct_clean}/{total_clean})")
    
    mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    print("\nPer-Class Accuracy on Clean Images:")
    for i, class_name in enumerate(mnist_classes):
        class_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"  Digit {class_name}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
    
    # Test on flow-interpolated images at various timesteps
    print("\n2. Performance on FLOW-INTERPOLATED images at various timesteps:")
    print("-" * 80)
    
    timesteps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    timestep_accuracies = []
    
    for t_val in timesteps:
        correct_t = 0
        total_t = 0
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.shape[0]
            
            # Create flow interpolation: x_t = (1-t) * noise + t * data
            x0 = torch.randn_like(images)
            t = torch.full((batch_size,), t_val, device=device)
            x_t = flow_matching.sample_xt(x0, images, t)
            
            with amp.autocast():
                logits = classifier(x_t, t)
            
            pred = logits.argmax(dim=-1)
            correct_t += (pred == labels).sum().item()
            total_t += batch_size
            
            # Only use subset of test set for speed
            if total_t >= 5000:
                break
        
        accuracy_t = correct_t / total_t
        timestep_accuracies.append((t_val, accuracy_t))
        print(f"  t={t_val:.1f}: Accuracy = {accuracy_t:.4f} ({correct_t}/{total_t})")
    
    # Test with pure noise (t=0.0) - should be close to random (10% for 10 classes)
    print("\n3. Sanity check on PURE NOISE (t=0.0):")
    print("-" * 80)
    
    correct_noise = 0
    total_noise = 0
    
    for _ in range(20):  # Test on 20 batches
        batch_size = 256
        x_noise = torch.randn(batch_size, 1, 28, 28, device=device)
        labels = torch.randint(0, 10, (batch_size,), device=device)
        t = torch.zeros(batch_size, device=device)
        
        with amp.autocast():
            logits = classifier(x_noise, t)
        
        pred = logits.argmax(dim=-1)
        correct_noise += (pred == labels).sum().item()
        total_noise += batch_size
    
    accuracy_noise = correct_noise / total_noise
    print(f"Accuracy on Pure Noise: {accuracy_noise:.4f} ({correct_noise}/{total_noise})")
    print(f"Expected (random): ~0.1000 for 10 classes")
    print(f"Difference from random: {abs(accuracy_noise - 0.1):.4f}")
    
    # Summary
    print("\n" + "="*80)
    print("CLASSIFIER EVALUATION SUMMARY")
    print("="*80)
    print(f"Clean Images (t=1.0):     {accuracy_clean:.4f}")
    print(f"Flow Interpolated (t=0.6): {[acc for t, acc in timestep_accuracies if t == 0.6][0]:.4f}")
    print(f"Pure Noise (t=0.0):       {accuracy_noise:.4f}")
    print()
    
    if accuracy_clean > 0.85:
        print("✓ Classifier is working WELL on clean images (>85% accuracy)")
    elif accuracy_clean > 0.7:
        print("⚠ Classifier is working REASONABLY on clean images (70-85% accuracy)")
    else:
        print("✗ Classifier may need more training (<70% accuracy on clean images)")
    
    if accuracy_noise < 0.15:
        print("✓ Classifier shows appropriate uncertainty on pure noise")
    else:
        print("⚠ Classifier may be overconfident on pure noise")
    
    print("="*80)
    
    return {
        'clean_accuracy': accuracy_clean,
        'per_class_accuracy': {mnist_classes[i]: class_correct[i] / class_total[i] 
                               for i in range(10) if class_total[i] > 0},
        'timestep_accuracies': timestep_accuracies,
        'noise_accuracy': accuracy_noise
    }

# ============================================================================
# SAMPLING FUNCTIONS
# ============================================================================

@torch.no_grad()
def sample_flow_ode(model, num_samples, num_steps=50, batch_size=50, use_heun=True, device='cuda', save_individual=False):
    """Generate exactly num_samples using ODE solver for MNIST"""
    
    model.eval()
    if hasattr(model, 'ema_model'):
        sample_fn = model.ema_model
    else:
        sample_fn = model
    
    all_samples = []
    samples_generated = 0
    
    if save_individual:
        os.makedirs(f'{OUTPUT_FOLDER}/generated_samples', exist_ok=True)
    
    pbar = tqdm(total=num_samples, desc="Generating unconditional samples")
    
    while samples_generated < num_samples:
        curr_batch_size = min(batch_size, num_samples - samples_generated)
        
        x = torch.randn(curr_batch_size, 1, 28, 28, device=device)
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = step * dt
            t_next = min((step + 1) * dt, 1.0)
            t_tensor = torch.full((curr_batch_size,), t, device=device)
            
            if use_heun and step < num_steps - 1:
                with amp.autocast():
                    v_t = sample_fn(x, t_tensor)
                x_euler = x + (t_next - t) * v_t
                
                t_next_tensor = torch.full((curr_batch_size,), t_next, device=device)
                with amp.autocast():
                    v_next = sample_fn(x_euler, t_next_tensor)
                
                x = x + (t_next - t) * 0.5 * (v_t + v_next)
            else:
                with amp.autocast():
                    v_t = sample_fn(x, t_tensor)
                x = x + (t_next - t) * v_t
        
        if save_individual:
            x_clamped = torch.clamp((x + 1) / 2, 0, 1)
            for i in range(curr_batch_size):
                sample_idx = samples_generated + i
                save_image(x_clamped[i], f'{OUTPUT_FOLDER}/generated_samples/sample_{sample_idx:04d}.png')
        
        all_samples.append(x.cpu())
        samples_generated += curr_batch_size
        pbar.update(curr_batch_size)
        
        torch.cuda.empty_cache()
    
    pbar.close()
    return torch.cat(all_samples, dim=0)

def sample_with_classifier_guidance(
    flow_model,
    classifier,
    num_samples,
    class_labels,
    num_steps=50,
    guidance_scale=2.0,
    use_heun=True,
    device='cuda'
):
    """Sample using classifier guidance for MNIST flow matching"""
    
    flow_model.eval()
    classifier.eval()
    
    if hasattr(flow_model, 'ema_model'):
        sample_fn = flow_model.ema_model
    else:
        sample_fn = flow_model
    
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    
    if isinstance(class_labels, int):
        class_labels = torch.full((num_samples,), class_labels, device=device)
    elif isinstance(class_labels, list):
        class_labels = torch.tensor(class_labels, device=device)
    
    dt = 1.0 / num_steps
    
    for step in range(num_steps):
        t = step * dt
        t_next = min((step + 1) * dt, 1.0)
        t_tensor = torch.full((num_samples,), t, device=device)
        
        # Enable gradients for classifier guidance
        x = x.detach().requires_grad_(True)
        
        logits = classifier(x, t_tensor)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(num_samples), class_labels]
        
        grad = torch.autograd.grad(selected.sum(), x)[0]
        x = x.detach()
        
        with torch.no_grad(), amp.autocast():
            v_uncond = sample_fn(x, t_tensor)
        
        # Add gradient to push toward the class
        v_guided = v_uncond + guidance_scale * grad
        
        if use_heun and step < num_steps - 1:
            x_euler = x + (t_next - t) * v_guided
            
            t_next_tensor = torch.full((num_samples,), t_next, device=device)
            
            x_euler = x_euler.detach().requires_grad_(True)
            logits_next = classifier(x_euler, t_next_tensor)
            log_probs_next = F.log_softmax(logits_next, dim=-1)
            selected_next = log_probs_next[range(num_samples), class_labels]
            grad_next = torch.autograd.grad(selected_next.sum(), x_euler)[0]
            x_euler = x_euler.detach()
            
            with torch.no_grad(), amp.autocast():
                v_next_uncond = sample_fn(x_euler, t_next_tensor)
            v_next_guided = v_next_uncond + guidance_scale * grad_next
            
            x = x + (t_next - t) * 0.5 * (v_guided + v_next_guided)
        else:
            x = x + (t_next - t) * v_guided
    
    return x

# ============================================================================
# SIMPLIFIED EVALUATION
# ============================================================================

def evaluate_model_simple(
    flow_model, 
    classifier,
    test_loader, 
    num_samples=5000,
    guidance_scales=[0.0, 1.0, 2.0, 4.0, 6.0],
    eval_batch_size=25
):
    """Simplified evaluation with better memory management"""
    
    print(f"\n{'='*70}")
    print(f"Model Evaluation - MNIST ({num_samples} samples)")
    print(f"{'='*70}")
    
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get real samples
    print("Loading real samples...")
    real_samples = []
    for images, _ in test_loader:
        real_samples.append(images)
        if len(torch.cat(real_samples)) >= num_samples:
            break
    real_samples = torch.cat(real_samples)[:num_samples]
    
    # Initialize feature extractor
    print("Initializing feature extractor...")
    fid_extractor = SimplifiedFIDExtractor(device)
    pr_evaluator = SimplifiedPrecisionRecall(k=3)
    
    results = {}
    
    # Evaluate unconditional generation
    print("\n" + "="*70)
    print("Evaluating Unconditional Generation")
    print("="*70)
    
    print(f"Generating {num_samples} unconditional samples...")
    uncond_samples = sample_flow_ode(
        flow_model, num_samples, num_steps=50, batch_size=eval_batch_size, 
        use_heun=True, save_individual=False
    )
    
    print("Computing metrics for unconditional samples...")
    
    # Extract features with aggressive batching
    print("  Extracting real features...")
    real_features = fid_extractor.extract_features_batch(real_samples, batch_size=8)
    
    print("  Extracting fake features...")
    fake_features = fid_extractor.extract_features_batch(uncond_samples, batch_size=8)
    
    # Compute metrics
    fid_uncond = calculate_fid_simple(real_features, fake_features)
    precision_uncond, recall_uncond = pr_evaluator.compute_simple_pr(real_features, fake_features)
    
    results['unconditional'] = {
        'fid': fid_uncond,
        'precision': precision_uncond,
        'recall': recall_uncond
    }
    
    print(f"Unconditional Results:")
    print(f"   FID: {fid_uncond:.2f}")
    print(f"   Precision: {precision_uncond:.4f}")
    print(f"   Recall: {recall_uncond:.4f}")
    
    # Clean up
    del uncond_samples, fake_features
    torch.cuda.empty_cache()
    gc.collect()
    
    # Evaluate classifier-guided generation
    print("\n" + "="*70)
    print("Evaluating Classifier-Guided Generation")
    print("="*70)
    
    for guidance_scale in guidance_scales:
        print(f"\nEvaluating guidance scale {guidance_scale}...")
        
        # Generate fewer samples per class for efficiency
        samples_per_class = num_samples // 10
        guided_samples = []
        
        for class_idx in range(10):
            print(f"  Generating class {class_idx}...")
            class_samples = sample_with_classifier_guidance(
                flow_model,
                classifier,
                num_samples=samples_per_class,
                class_labels=class_idx,
                num_steps=50,
                guidance_scale=guidance_scale,
                use_heun=True
            )
            guided_samples.append(class_samples.cpu())
            torch.cuda.empty_cache()
        
        guided_samples = torch.cat(guided_samples)
        
        print("  Computing metrics...")
        guided_features = fid_extractor.extract_features_batch(guided_samples, batch_size=8)
        
        fid_guided = calculate_fid_simple(real_features, guided_features)
        precision_guided, recall_guided = pr_evaluator.compute_simple_pr(real_features, guided_features)
        
        results[f'guided_{guidance_scale}'] = {
            'fid': fid_guided,
            'precision': precision_guided,
            'recall': recall_guided
        }
        
        print(f"Guidance Scale {guidance_scale} Results:")
        print(f"   FID: {fid_guided:.2f}")
        print(f"   Precision: {precision_guided:.4f}")
        print(f"   Recall: {recall_guided:.4f}")
        
        # Clean up
        del guided_samples, guided_features
        torch.cuda.empty_cache()
        gc.collect()
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_samples(model, classifier, num_samples=64):
    """Visualize generated samples for MNIST"""
    
    guidance_scales = [0.0, 1.0, 2.0, 4.0, 6.0]
    
    os.makedirs(f'{OUTPUT_FOLDER}/samples', exist_ok=True)
    
    print("\nGenerating visualization samples...")
    
    # Unconditional samples
    print("Generating unconditional samples...")
    uncond_samples = sample_flow_ode(
        model, num_samples, num_steps=50, batch_size=32, use_heun=True
    )
    grid = make_grid(uncond_samples, nrow=8, normalize=True, value_range=(-1, 1))
    save_image(grid, f'{OUTPUT_FOLDER}/samples/unconditional_samples.png')
    
    # Guided samples
    for guidance_scale in guidance_scales:
        print(f"Generating guided samples (scale={guidance_scale})...")
        
        class_labels = [i % 10 for i in range(num_samples)]
        
        samples = sample_with_classifier_guidance(
            model, classifier, num_samples, 
            class_labels=class_labels,
            num_steps=50, guidance_scale=guidance_scale, use_heun=True
        )
        
        grid = make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
        save_image(grid, f'{OUTPUT_FOLDER}/samples/guided_scale_{guidance_scale}.png')
    
    print(f"All samples saved to '{OUTPUT_FOLDER}/samples/' directory")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("="*80)
    print("MNIST Flow Matching with Classifier Guidance (FIXED)")
    print(f"TRAINING: {NUM_EPOCHS_FLOW} epochs | EVALUATION: {NUM_SAMPLES_EVAL} samples")
    print(f"OUTPUT FOLDER: {OUTPUT_FOLDER}")
    print("="*80)
    
    # Configuration
    BATCH_SIZE = 128
    PHYSICAL_BATCH_SIZE = 32
    NUM_SAMPLES_SAVE = 100
    EVAL_BATCH_SIZE = 25  # Reduced for memory efficiency
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("\nLoading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, 
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    # Initialize models
    print("\nInitializing models...")
    flow_model = UNetModelMNIST(
        image_size=28,
        in_channels=1,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(14,),
        dropout=0.0,
        channel_mult=(1, 2, 2),
        use_scale_shift_norm=True,
        resblock_updown=True,
    ).to(device)
    
    print(f"Flow model parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    
    # Train or load flow model
    flow_checkpoint_path = f'{OUTPUT_FOLDER}/checkpoints/flow_final.pt'
    if os.path.exists(flow_checkpoint_path):
        print("Loading pre-trained flow model...")
        checkpoint = torch.load(flow_checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            flow_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try loading as EMA model
            flow_model = EMA(flow_model)
            flow_model.ema_model.load_state_dict(checkpoint)
    else:
        print(f"\nTraining flow model for {NUM_EPOCHS_FLOW} epochs...")
        flow_model, _ = train_flow_matching(
            flow_model, train_loader, num_epochs=NUM_EPOCHS_FLOW,
            batch_size=BATCH_SIZE, physical_batch_size=PHYSICAL_BATCH_SIZE,
        )
        os.makedirs(f'{OUTPUT_FOLDER}/checkpoints', exist_ok=True)
        torch.save({
            'model_state_dict': flow_model.ema_model.state_dict() if hasattr(flow_model, 'ema_model') else flow_model.state_dict()
        }, flow_checkpoint_path)
    
    # Initialize simplified classifier
    print("\nInitializing classifier...")
    classifier = SimplifiedMNISTClassifier(
        num_classes=10, 
        time_emb_dim=128,
        base_channels=64,
        dropout=0.1
    ).to(device)
    
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    
    classifier_checkpoint_path = f'{OUTPUT_FOLDER}/checkpoints/classifier_simple_final.pt'
    
    if os.path.exists(classifier_checkpoint_path):
        print("Loading pre-trained classifier...")
        classifier.load_state_dict(torch.load(classifier_checkpoint_path, map_location=device))
    else:
        print(f"\nTraining classifier for {NUM_EPOCHS_CLASSIFIER} epochs...")
        classifier, _ = train_flow_classifier(
            classifier, flow_model, train_loader, num_epochs=NUM_EPOCHS_CLASSIFIER, warmup_epochs=5
        )
        os.makedirs(f'{OUTPUT_FOLDER}/checkpoints', exist_ok=True)
        torch.save(classifier.state_dict(), classifier_checkpoint_path)
    
    # Generate and save individual samples
    print("\n" + "="*80)
    print(f"GENERATING {NUM_SAMPLES_SAVE} INDIVIDUAL SAMPLES")
    print("="*80)
    
    saved_samples = sample_flow_ode(
        flow_model, NUM_SAMPLES_SAVE, num_steps=50, batch_size=25, 
        use_heun=True, save_individual=True
    )
    print(f"Saved {NUM_SAMPLES_SAVE} individual samples to '{OUTPUT_FOLDER}/generated_samples/'")
    
    # Comprehensive evaluation
    print("\n" + "="*80)
    print(f"EVALUATION ON {NUM_SAMPLES_EVAL} SAMPLES")
    print("="*80)
    
    results = evaluate_model_simple(
        flow_model, classifier, test_loader, 
        num_samples=NUM_SAMPLES_EVAL,
        guidance_scales=[0.0,1.0, 2.0, 4.0 ,6.0],
        eval_batch_size=EVAL_BATCH_SIZE
    )
    
    # Evaluate classifier performance
    print("\n" + "="*80)
    print("EVALUATING CLASSIFIER PERFORMANCE")
    print("="*80)
    
    classifier_performance = evaluate_classifier_performance(
        classifier, flow_model, test_loader, device=device
    )
    
    # Visualize samples
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    visualize_samples(flow_model, classifier, num_samples=64)
    
    # Save results
    print("\nSaving results...")
    os.makedirs(f'{OUTPUT_FOLDER}/results', exist_ok=True)
    with open(f'{OUTPUT_FOLDER}/results/evaluation_results.txt', 'w') as f:
        f.write(f"MNIST Flow Matching - Evaluation Results\n")
        f.write(f"Training: {NUM_EPOCHS_FLOW} epochs | Evaluation: {NUM_SAMPLES_EVAL} samples\n")
        f.write("="*70 + "\n\n")
        
        # Add classifier performance section
        f.write("CLASSIFIER PERFORMANCE:\n")
        f.write(f"   Clean Images Accuracy: {classifier_performance['clean_accuracy']:.4f}\n")
        f.write(f"   Pure Noise Accuracy: {classifier_performance['noise_accuracy']:.4f}\n")
        f.write("\n   Per-Class Accuracy on Clean Images:\n")
        for class_name, acc in classifier_performance['per_class_accuracy'].items():
            f.write(f"      Digit {class_name}: {acc:.4f}\n")
        f.write("\n   Accuracy at Different Timesteps:\n")
        for t_val, acc in classifier_performance['timestep_accuracies']:
            f.write(f"      t={t_val:.1f}: {acc:.4f}\n")
        f.write("\n" + "="*70 + "\n\n")
        
        # Original metrics
        for key, metrics in results.items():
            f.write(f"{key}:\n")
            f.write(f"   FID: {metrics['fid']:.2f}\n")
            f.write(f"   Precision: {metrics['precision']:.4f}\n")
            f.write(f"   Recall: {metrics['recall']:.4f}\n\n")
    
    # Print final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    \
    print("CLASSIFIER PERFORMANCE:")
    print(f"  Clean Images Accuracy: {classifier_performance['clean_accuracy']:.4f}")
    print(f"  Pure Noise Accuracy:   {classifier_performance['noise_accuracy']:.4f}")
    print()
    
    for key, metrics in results.items():
        print(f"{key}:")
        print(f"   FID: {metrics['fid']:.2f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print()
    
    print("="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Results saved to '{OUTPUT_FOLDER}/results/evaluation_results.txt'")
    print(f"Visualizations saved to '{OUTPUT_FOLDER}/samples/'")
    print(f"Checkpoints saved to '{OUTPUT_FOLDER}/checkpoints/'")
    
    return flow_model, classifier, results

if __name__ == "__main__":
    flow_model, classifier, results = main()
