"""
DALLE: Autoregressive transformer for text-to-ECG generation.

This module provides:
- DALLE: Base autoregressive model for text-to-ECG
- DALLE_concat: Variant with age/gender conditioning via concatenation
- CLIP: Contrastive model for ECG-text alignment (optional)
"""

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from text_to_ecg.models.vae import DiscreteVAE, VQVAE
from text_to_ecg.models.transformer import Transformer, DivideMax
from text_to_ecg.utils.helpers import exists, default, eval_decorator, set_requires_grad, top_k


class always:
    """Helper class that always returns the same value."""

    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return self.val


def masked_mean(t, mask, dim=1):
    """Compute mean of tensor with mask."""
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


class DALLE(nn.Module):
    """Autoregressive transformer for text-to-ECG generation.

    Args:
        dim: Model dimension
        vae: Pretrained VQ-VAE for ECG tokenization
        num_text_tokens: Vocabulary size for text
        text_seq_len: Maximum text sequence length
        depth: Number of transformer layers
        heads: Number of attention heads
        dim_head: Dimension per head
        reversible: Use reversible layers
        attn_dropout: Attention dropout rate
        ff_dropout: Feed-forward dropout rate
        sparse_attn: Use sparse attention
        attn_types: Attention types to use
        loss_img_weight: Weight for ECG token prediction loss
        stable: Use numerically stable operations
        sandwich_norm: Use sandwich normalization
        shift_tokens: Use token shifting
        rotary_emb: Use rotary positional embeddings
    """

    def __init__(
        self,
        *,
        dim,
        vae,
        num_text_tokens=4000,
        text_seq_len=256,
        depth,
        heads=8,
        dim_head=64,
        reversible=False,
        attn_dropout=0.,
        ff_dropout=0,
        sparse_attn=False,
        attn_types=None,
        loss_img_weight=7,
        stable=False,
        sandwich_norm=False,
        shift_tokens=True,
        rotary_emb=True
    ):
        super().__init__()
        assert isinstance(vae, (DiscreteVAE, VQVAE)), 'vae must be an instance of DiscreteVAE or VQVAE'

        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = 312  # ECG token sequence length after VAE encoding
        image_seq_len = 312

        num_text_tokens = num_text_tokens + text_seq_len  # Reserve padding tokens

        self.text_emb = nn.Embedding(num_text_tokens, dim)
        self.image_emb = nn.Embedding(num_image_tokens, dim)

        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim) if not rotary_emb else always(0)
        self.image_pos_emb = nn.Embedding(image_fmap_size, dim) if not rotary_emb else always(0)

        self.num_text_tokens = num_text_tokens
        self.num_image_tokens = num_image_tokens

        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        seq_len = text_seq_len + image_seq_len
        total_tokens = num_text_tokens + num_image_tokens
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len

        self.vae = vae
        set_requires_grad(self.vae, False)  # Freeze VAE

        self.transformer = Transformer(
            dim=dim,
            causal=True,
            seq_len=seq_len,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            reversible=reversible,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            attn_types=attn_types,
            image_fmap_size=image_fmap_size,
            sparse_attn=sparse_attn,
            stable=stable,
            sandwich_norm=sandwich_norm,
            shift_tokens=shift_tokens,
            rotary_emb=rotary_emb
        )

        self.stable = stable
        if stable:
            self.norm_by_max = DivideMax(dim=-1)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),
        )

        # Create logits mask for autoregressive prediction
        # Position i predicts token at position i+1
        # Positions 0 to text_seq_len-2 predict text tokens (positions 1 to text_seq_len-1)
        # Positions text_seq_len-1 and beyond predict image tokens
        seq_range = torch.arange(seq_len)
        logits_range = torch.arange(total_tokens)

        seq_range = rearrange(seq_range, 'n -> () n ()')
        logits_range = rearrange(logits_range, 'd -> () () d')

        # text_seq_len - 1 is the boundary: it predicts the first image token
        logits_mask = (
            ((seq_range >= text_seq_len - 1) & (logits_range < num_text_tokens)) |
            ((seq_range < text_seq_len - 1) & (logits_range >= num_text_tokens))
        )

        self.register_buffer('logits_mask', logits_mask, persistent=False)
        self.loss_img_weight = loss_img_weight

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        text,
        *,
        clip=None,
        mask=None,
        filter_thres=0.5,
        decoder_model=None,
        temperature=1.,
        img=None,
        num_init_img_tokens=None
    ):
        """Generate ECG from text.

        Args:
            text: Text token tensor [B, T]
            clip: Optional CLIP model for reranking
            mask: Attention mask
            filter_thres: Top-k threshold
            decoder_model: 'vqvae' or 'hifi' for decoding
            temperature: Sampling temperature
            img: Optional initial ECG for priming
            num_init_img_tokens: Number of initial image tokens

        Returns:
            Generated ECG tensor [B, C, T]
        """
        vae = self.vae
        text_seq_len = self.text_seq_len
        image_seq_len = self.image_seq_len
        num_text_tokens = self.num_text_tokens
        total_len = text_seq_len + image_seq_len

        text = text[:, :text_seq_len]
        out = text

        if exists(img):
            indices = vae.get_codebook_indices(img)
            num_img_tokens = default(num_init_img_tokens, int(0.4375 * image_seq_len))
            assert num_img_tokens < image_seq_len
            indices = indices[:, :num_img_tokens]
            out = torch.cat((out, indices), dim=-1)

        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len

            text, image = out[:, :text_seq_len], out[:, text_seq_len:]
            logits = self(text, image, mask=mask)[:, -1, :]

            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            sample -= (num_text_tokens if is_image else 0)
            out = torch.cat((out, sample), dim=-1)

            if out.shape[1] <= text_seq_len:
                mask = F.pad(mask, (0, 1), value=True)

        img_seq = out[:, -image_seq_len:]

        # Decode using VQ-VAE or HiFi-GAN
        if decoder_model == "vqvae":
            images = vae.decode(img_seq)
        else:
            # HiFi-GAN decoder - need to be loaded externally
            from text_to_ecg.models.hifi_gan import CodeGenerator
            raise NotImplementedError(
                "HiFi-GAN decoder must be loaded manually. "
                "Use vae.decode() or load CodeGenerator separately."
            )

        if exists(clip):
            text_seq = out[:, :text_seq_len]
            scores = clip(text_seq, images, return_loss=False)
            return images, scores

        return images

    def forward(self, text, image=None, mask=None, return_loss=False):
        """Forward pass through DALLE.

        Args:
            text: Text tokens [B, T_text]
            image: ECG tokens [B, T_ecg] (optional for training)
            mask: Attention mask
            return_loss: Return cross-entropy loss

        Returns:
            Logits or loss depending on return_loss flag
        """
        assert text.shape[-1] == self.text_seq_len, \
            f'text length {text.shape[-1]} != expected {self.text_seq_len}'
        device = text.device
        total_seq_len = self.total_seq_len

        # Replace padding with unique padding tokens
        text_range = torch.arange(self.text_seq_len, device=device) + \
            (self.num_text_tokens - self.text_seq_len)
        text = torch.where(text == 0, text_range, text)

        # Get image tokens if not provided
        if exists(image) and not torch.is_tensor(image):
            raise ValueError("image must be a tensor of codebook indices")

        # Prepare text embeddings
        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device=device))

        if exists(image):
            image_emb = self.image_emb(image)
            image_emb += self.image_pos_emb(torch.arange(image.shape[1], device=device))
            tokens = torch.cat((text_emb, image_emb), dim=1)
        else:
            tokens = text_emb

        # Create mask if needed
        if exists(mask):
            mask = F.pad(mask, (0, tokens.shape[1] - mask.shape[1]), value=True)

        # Transform
        out = self.transformer(tokens, mask=mask)

        if self.stable:
            out = self.norm_by_max(out)

        logits = self.to_logits(out)

        # Mask logits
        mask = self.logits_mask[:, :tokens.shape[1]]
        max_neg_value = -torch.finfo(logits.dtype).max
        logits.masked_fill_(mask, max_neg_value)

        if not return_loss:
            return logits

        # Calculate loss
        assert exists(image), 'when training, image tokens must be provided'

        # Shift for autoregressive loss
        logits = logits[:, :-1]
        labels = torch.cat((text[:, 1:], image + self.num_text_tokens), dim=-1)

        # Compute loss
        logits = rearrange(logits, 'b n c -> b c n')
        text_logits = logits[:, :, :self.text_seq_len - 1]
        image_logits = logits[:, :, self.text_seq_len - 1:]

        text_labels = labels[:, :self.text_seq_len - 1]
        image_labels = labels[:, self.text_seq_len - 1:]

        text_loss = F.cross_entropy(text_logits, text_labels)
        image_loss = F.cross_entropy(image_logits, image_labels)

        loss = (text_loss + self.loss_img_weight * image_loss) / (1 + self.loss_img_weight)

        return loss


class DALLE_concat(DALLE):
    """DALLE with age/gender conditioning via concatenation.

    Age and gender are prepended to the text sequence as special tokens
    before feeding to the transformer.

    Age is discretized into 10-year bins (0-9, 10-19, ..., 90+) = 10 bins
    Gender is encoded as 0 (female) or 1 (male)

    Special token indices are assigned at the end of the text vocabulary.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Reserve special tokens for age (10 bins) and gender (2 values)
        # These are mapped to indices at the end of the text token range
        self.num_age_bins = 10
        self.num_gender_tokens = 2
        # Age tokens: num_text_tokens - 12, ..., num_text_tokens - 3
        # Gender tokens: num_text_tokens - 2, num_text_tokens - 1
        self.age_token_offset = self.num_text_tokens - self.text_seq_len - 12
        self.gender_token_offset = self.num_text_tokens - self.text_seq_len - 2

    def _encode_age(self, age):
        """Convert age to discrete token index."""
        # Discretize age into 10-year bins: 0-9 -> 0, 10-19 -> 1, ..., 90+ -> 9
        age_bin = torch.clamp(age.long() // 10, min=0, max=self.num_age_bins - 1)
        return age_bin + self.age_token_offset

    def _encode_gender(self, gender):
        """Convert gender to token index."""
        # Gender: 0 (female), 1 (male)
        return gender.long() + self.gender_token_offset

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        text,
        age=None,
        gender=None,
        *,
        clip=None,
        mask=None,
        filter_thres=0.5,
        decoder_model=None,
        temperature=1.,
        img=None,
        num_init_img_tokens=None
    ):
        """Generate ECG from text with age/gender conditioning.

        Args:
            text: Text token tensor [B, T]
            age: Age tensor [B]
            gender: Gender tensor [B]
            clip: Optional CLIP model for reranking
            mask: Attention mask
            filter_thres: Top-k threshold
            decoder_model: 'vqvae' or 'hifi' for decoding
            temperature: Sampling temperature
            img: Optional initial ECG for priming
            num_init_img_tokens: Number of initial image tokens

        Returns:
            Generated ECG tensor [B, C, T]
        """
        vae = self.vae
        text_seq_len = self.text_seq_len
        image_seq_len = self.image_seq_len
        num_text_tokens = self.num_text_tokens
        total_len = text_seq_len + image_seq_len
        device = text.device

        # Concatenate age/gender tokens if provided
        text = text[:, :text_seq_len - 2]  # Leave room for age/gender
        if exists(age) and exists(gender):
            age_tokens = self._encode_age(age).unsqueeze(1).to(device)
            gender_tokens = self._encode_gender(gender).unsqueeze(1).to(device)
            text = torch.cat([gender_tokens, age_tokens, text], dim=1)

        text = text[:, :text_seq_len]
        out = text

        if exists(img):
            indices = vae.get_codebook_indices(img)
            num_img_tokens = default(num_init_img_tokens, int(0.4375 * image_seq_len))
            assert num_img_tokens < image_seq_len
            indices = indices[:, :num_img_tokens]
            out = torch.cat((out, indices), dim=-1)

        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len

            text_part, image_part = out[:, :text_seq_len], out[:, text_seq_len:]
            logits = self.forward_generate(text_part, image_part, mask=mask)[:, -1, :]

            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            sample -= (num_text_tokens if is_image else 0)
            out = torch.cat((out, sample), dim=-1)

            if out.shape[1] <= text_seq_len and exists(mask):
                mask = F.pad(mask, (0, 1), value=True)

        img_seq = out[:, -image_seq_len:]

        # Decode with VQ-VAE (for HiFi-GAN decoding, use generate.py --decoder hifi)
        images = vae.decode(img_seq)

        if exists(clip):
            text_seq = out[:, :text_seq_len]
            scores = clip(text_seq, images, return_loss=False)
            return images, scores

        return images

    def forward_generate(self, text, image=None, mask=None):
        """Forward pass for generation (no loss computation)."""
        device = text.device

        # Replace padding
        text_range = torch.arange(self.text_seq_len, device=device) + \
            (self.num_text_tokens - self.text_seq_len)
        text = torch.where(text == 0, text_range, text)

        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device=device))

        if exists(image) and image.shape[1] > 0:
            image_emb = self.image_emb(image)
            image_emb += self.image_pos_emb(torch.arange(image.shape[1], device=device))
            tokens = torch.cat((text_emb, image_emb), dim=1)
        else:
            tokens = text_emb

        out = self.transformer(tokens, mask=mask)

        if self.stable:
            out = self.norm_by_max(out)

        logits = self.to_logits(out)

        mask = self.logits_mask[:, :tokens.shape[1]]
        max_neg_value = -torch.finfo(logits.dtype).max
        logits.masked_fill_(mask, max_neg_value)

        return logits

    def forward(self, text, image=None, age=None, gender=None, mask=None, return_loss=False):
        """Forward pass with age/gender conditioning.

        Args:
            text: Text tokens [B, T_text]
            image: ECG (raw signal) for training [B, C, T]
            age: Age tensor [B]
            gender: Gender tensor [B]
            mask: Attention mask
            return_loss: Return cross-entropy loss

        Returns:
            Logits or loss
        """
        device = text.device

        # Concatenate age/gender as discrete tokens
        text = text[:, :self.text_seq_len - 2]
        if exists(age) and exists(gender):
            age_tokens = self._encode_age(age).unsqueeze(1).to(device)
            gender_tokens = self._encode_gender(gender).unsqueeze(1).to(device)
            text = torch.cat([gender_tokens, age_tokens, text], dim=1)

        # Pad text to expected length
        if text.shape[1] < self.text_seq_len:
            text = F.pad(text, (0, self.text_seq_len - text.shape[1]))
        text = text[:, :self.text_seq_len]

        # Get image tokens
        if exists(image):
            image_tokens = self.vae.get_codebook_indices(image)
        else:
            image_tokens = None

        return super().forward(text, image_tokens, mask=mask, return_loss=return_loss)


class CLIP(nn.Module):
    """Contrastive model for ECG-text alignment.

    Args:
        vae: Pretrained VAE for ECG encoding
        dim_text: Text encoder dimension
        dim_image: Visual encoder dimension
        dim_latent: Latent embedding dimension
        num_text_tokens: Text vocabulary size
        text_enc_depth: Text transformer depth
        text_seq_len: Text sequence length
        visual_seq_len: Visual sequence length
        text_heads: Text attention heads
        num_visual_tokens: Number of visual tokens
        visual_enc_depth: Visual transformer depth
        visual_heads: Visual attention heads
        attn_dropout: Attention dropout
        ff_dropout: Feed-forward dropout
    """

    def __init__(
        self,
        *,
        vae,
        dim_text=512,
        dim_image=512,
        dim_latent=512,
        num_text_tokens=1000,
        text_enc_depth=6,
        text_seq_len=105,
        visual_seq_len=312,
        text_heads=8,
        num_visual_tokens=1024,
        visual_enc_depth=6,
        visual_heads=8,
        attn_dropout=0,
        ff_dropout=0
    ):
        super().__init__()
        self.text_emb = nn.Embedding(num_text_tokens, dim_text)
        self.text_pos_emb = nn.Embedding(text_seq_len, dim_text)
        self.text_transformer = Transformer(
            causal=False, seq_len=text_seq_len, dim=dim_text, depth=text_enc_depth,
            heads=text_heads, rotary_emb=False, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias=False)

        self.visual_emb = nn.Embedding(num_visual_tokens, dim_image)
        self.visual_pos_emb = nn.Embedding(visual_seq_len, dim_image)
        self.visual_transformer = Transformer(
            causal=False, seq_len=visual_seq_len, dim=dim_image, depth=visual_enc_depth,
            heads=visual_heads, rotary_emb=False
        )
        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias=False)

        self.temperature = nn.Parameter(torch.tensor(1.))
        self.vae = vae

    def forward(self, text, image, text_mask=None, return_loss=False):
        """Compute CLIP similarity between text and ECG.

        Args:
            text: Text tokens [B, T]
            image: ECG signal [B, C, T]
            text_mask: Attention mask for text
            return_loss: Return contrastive loss

        Returns:
            Similarity scores or contrastive loss
        """
        b, device = text.shape[0], text.device

        # Encode image through VAE
        image = self.vae.encoder(image).transpose(1, 2)

        # Text encoding
        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device=device))
        enc_text = self.text_transformer(text_emb, mask=text_mask)

        if exists(text_mask):
            text_latents = masked_mean(enc_text, text_mask, dim=1)
        else:
            text_latents = enc_text.mean(dim=1)

        image_latents = image.mean(dim=1)

        text_latents = self.to_text_latent(text_latents)
        image_latents = self.to_visual_latent(image_latents)

        # Normalize
        text_latents, image_latents = map(
            lambda t: F.normalize(t, p=2, dim=-1),
            (text_latents, image_latents)
        )

        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum('n d, n d -> n', text_latents, image_latents) * temp
            return sim / temp

        sim = einsum('i d, j d -> i j', text_latents, image_latents) * temp
        labels = torch.arange(b, device=device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        return loss
