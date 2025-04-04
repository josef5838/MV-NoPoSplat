from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from torch import nn

from .croco.blocks import DecoderBlock
from .croco.croco import CroCoNet
from .croco.misc import fill_default_args, freeze_all_params, transpose_to_landscape, is_symmetrized, interleave, \
    make_batch_symmetric
from .croco.patch_embed import get_patch_embed
from .backbone import Backbone
from ....geometry.camera_emb import get_intrinsic_embedding

inf = float('inf')


croco_params = {
    'ViTLarge_BaseDecoder': {
        'enc_depth': 24,
        'dec_depth': 12,
        'enc_embed_dim': 1024,
        'dec_embed_dim': 768,
        'enc_num_heads': 16,
        'dec_num_heads': 12,
        'pos_embed': 'RoPE100',
        'img_size': (512, 512),
    },
}

default_dust3r_params = {
    'enc_depth': 24,
    'dec_depth': 12,
    'enc_embed_dim': 1024,
    'dec_embed_dim': 768,
    'enc_num_heads': 16,
    'dec_num_heads': 12,
    'pos_embed': 'RoPE100',
    'patch_embed_cls': 'PatchEmbedDust3R',
    'img_size': (512, 512),
    'head_type': 'dpt',
    'output_mode': 'pts3d',
    'depth_mode': ('exp', -inf, inf),
    'conf_mode': ('exp', 1, inf)
}

@dataclass
class BackboneFast3rCfg:
    name: Literal["fast3r", "fast3r_multi"]
    model: Literal["ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"]  # keep interface for the last two models, but they are not supported
    patch_embed_cls: str = 'PatchEmbedDust3R'  # PatchEmbedDust3R or ManyAR_PatchEmbed
    asymmetry_decoder: bool = True
    intrinsics_embed_loc: Literal["encoder", "decoder", "none"] = 'none'
    intrinsics_embed_degree: int = 0
    intrinsics_embed_type: Literal["pixelwise", "linear", "token"] = 'token'  # linear or dpt

class Fast3rMulti(CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self, cfg: BackboneFast3rCfg, d_in: int) -> None:

        self.intrinsics_embed_loc = cfg.intrinsics_embed_loc
        self.intrinsics_embed_degree = cfg.intrinsics_embed_degree
        self.intrinsics_embed_type = cfg.intrinsics_embed_type
        self.intrinsics_embed_encoder_dim = 0
        self.intrinsics_embed_decoder_dim = 0
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_encoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3
        elif self.intrinsics_embed_loc == 'decoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_decoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3

        self.patch_embed_cls = cfg.patch_embed_cls
        self.croco_args = fill_default_args(croco_params[cfg.model], CroCoNet.__init__)

        super().__init__(**croco_params[cfg.model])

        if cfg.asymmetry_decoder:
            self.dec_blocks2 = deepcopy(self.dec_blocks)  # This is used in DUSt3R and MASt3R

        if self.intrinsics_embed_type == 'linear' or self.intrinsics_embed_type == 'token':
            self.intrinsic_encoder = nn.Linear(9, 1024)

        # self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        # self.set_freeze(freeze)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3):
        in_chans = in_chans + self.intrinsics_embed_encoder_dim # (4+1)**2 + 3
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans)

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        enc_embed_dim = enc_embed_dim + self.intrinsics_embed_decoder_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        assert freeze in ['none', 'mask', 'encoder'], f"unexpected freeze={freeze}"
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_decoder':  [self.mask_token, self.patch_embed, self.enc_blocks, self.enc_norm, self.decoder_embed, self.dec_blocks, self.dec_blocks2, self.dec_norm],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def _encode_image(self, image, true_shape, intrinsics_embed=None):
        # embed the image into patches  (x has size B x Npatches x C)
        # x = [batch, patch, channel]
        # pos = [batch patch 2]
        x, pos = self.patch_embed(image, true_shape=true_shape)

        if intrinsics_embed is not None:

            if self.intrinsics_embed_type == 'linear':
                x = x + intrinsics_embed
            elif self.intrinsics_embed_type == 'token':
                # intrinsics_embed BxV 1 1024
                # x -> BxV patch+1 1024
                # pose -> BxV patch+1 2
                x = torch.cat((x, intrinsics_embed), dim=1)
                add_pose = pos[:, 0:1, :].clone() #(BxV, 1, 2)
                add_pose[:, :, 0] += (pos[:, -1, 0].unsqueeze(-1) + 1) #
                pos = torch.cat((pos, add_pose), dim=1)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None
    
    def _get_random_image_pos(self, encoded_feats, batch_size, num_views, max_image_idx, device):
        """
        Generates non-repeating random image indices for each sample, retrieves corresponding
        positional embeddings for each view, and concatenates them.

        Args:
            encoded_feats (list of tensors): Encoded features for each view.
            batch_size (int): Number of samples in the batch.
            num_views (int): Number of views per sample.
            max_image_idx (int): Maximum image index for embedding.
            device (torch.device): Device to move data to.

        Returns:
            Tensor: Concatenated positional embeddings for the entire batch.
        """
        # Generate random non-repeating image IDs (on CPU)
        image_ids = torch.zeros(batch_size, num_views, dtype=torch.long)

        # First view is always 0 for all samples
        image_ids[:, 0] = 0

        # Get a generator that is unique to each rank, while also being deterministic based on the global across numbers of forward passes
        per_rank_generator = self._generate_per_rank_generator()

        # Generate random non-repeating IDs for the remaining views using the generator
        for b in range(batch_size):
            # Use the torch.Generator for randomness to ensure randomness between forward passes
            random_ids = torch.randperm(max_image_idx, generator=per_rank_generator)[:num_views - 1] + 1
            image_ids[b, 1:] = random_ids

        # Move the image IDs to the correct device
        image_ids = image_ids.to(device)

        # Initialize list to store positional embeddings for all views
        image_pos_list = []

        for i in range(num_views):
            # Retrieve the number of patches for this view
            num_patches = encoded_feats[i].shape[1]

            # Gather the positional embeddings for the entire batch based on the random image IDs
            image_pos_for_view = self.image_idx_emb[image_ids[:, i]]  # (B, D)

            # Expand the positional embeddings to match the number of patches
            image_pos_for_view = image_pos_for_view.unsqueeze(1).repeat(1, num_patches, 1)

            image_pos_list.append(image_pos_for_view)

        # Concatenate positional embeddings for all views along the patch dimension
        image_pos = torch.cat(image_pos_list, dim=1)  # (B, Npatches_total, D)

        return image_pos

    def _decoder_fast3r(self,  encoded_feats, positions, image_ids):

        # encoded_feats: 2 B L C
        # x: B 2*L C
        x = torch.cat(encoded_feats, dim=1)  # concate along the patch dimension
        # pos: B V*L 2
        pos = torch.cat(positions, dim=1)

        final_output = [x]  # before projection

        x = self.decoder_embed(x)

        # Generate random positional embeddings for all views and samples
        image_pos = self._get_random_image_pos(encoded_feats=encoded_feats,
                                                batch_size=encoded_feats[0].shape[0],
                                                num_views=len(encoded_feats),
                                                max_image_idx=self.image_idx_emb.shape[0] - 1,
                                                device=x.device)
        
        x += image_pos  # x has size B x Npatches x D, image_pos has size Npatches x D, so this is broadcasting
        
        for blk in self.dec_blocks:
            x = blk(x, pos)
            final_output.append(x)
        
        # 


    def _decoder(self, feat, pose, extra_embed=None):
        #feat: batch views patch channel
        #feat: b v l c
        b, v, l, c = feat.shape
        final_output = [feat]  # before projection
        if extra_embed is not None:
            feat = torch.cat((feat, extra_embed), dim=-1)

        # project to decoder dim
        f = rearrange(feat, "b v l c -> (b v) l c")
        f = self.decoder_embed(f)
        f = rearrange(f, "(b v) l c -> b v l c", b=b, v=v)
        final_output.append(f)

        def generate_ctx_views(x):
            b, v, l, c = x.shape
            ctx_views = x.unsqueeze(1).expand(b, v, v, l, c)
            mask = torch.arange(v).unsqueeze(0) != torch.arange(v).unsqueeze(1)
            ctx_views = ctx_views[:, mask].reshape(b, v, v - 1, l, c)  # B, V, V-1, L, C
            ctx_views = ctx_views.flatten(2, 3)  # B, V, (V-1)*L, C
            return ctx_views.contiguous()

        pos_ctx = generate_ctx_views(pose)
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            feat_current = final_output[-1]
            feat_current_ctx = generate_ctx_views(feat_current)
            # img1 side
            f1, _ = blk1(feat_current[:, 0].contiguous(), feat_current_ctx[:, 0].contiguous(), pose[:, 0].contiguous(), pos_ctx[:, 0].contiguous())
            f1 = f1.unsqueeze(1)
            # img2 side
            f2, _ = blk2(rearrange(feat_current[:, 1:], "b v l c -> (b v) l c"),
                         rearrange(feat_current_ctx[:, 1:], "b v l c -> (b v) l c"),
                         rearrange(pose[:, 1:], "b v l c -> (b v) l c"),
                         rearrange(pos_ctx[:, 1:], "b v l c -> (b v) l c"))
            f2 = rearrange(f2, "(b v) l c -> b v l c", b=b, v=v-1)
            # store the result
            final_output.append(torch.cat((f1, f2), dim=1))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        last_feat = rearrange(final_output[-1], "b v l c -> (b v) l c")
        last_feat = self.dec_norm(last_feat)
        final_output[-1] = rearrange(last_feat, "(b v) l c -> b v l c", b=b, v=v)

        return final_output

    def forward(self,
                context: dict,
                symmetrize_batch=False,
                return_views=False,
                ):
        b, v, _, h, w = context["image"].shape
        images_all = context["image"]

        # camera embedding in the encoder
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            intrinsic_embedding = get_intrinsic_embedding(context, degree=self.intrinsics_embed_degree)
            images_all = torch.cat((images_all, intrinsic_embedding), dim=2)

        intrinsic_embedding_all = None
        if self.intrinsics_embed_loc == 'encoder' and (self.intrinsics_embed_type == 'token' or self.intrinsics_embed_type == 'linear'):
            intrinsic_embedding = self.intrinsic_encoder(context["intrinsics"].flatten(2)) # B V 3 3 -》 B V 9 -> B V 1024
            intrinsic_embedding_all = rearrange(intrinsic_embedding, "b v c -> (b v) c").unsqueeze(1) # B V 1024 -> BxV 1 1024

        # step 1: encoder input images
        images_all = rearrange(images_all, "b v c h w -> (b v) c h w")
        # images_all.shape[-2:]: H W
        shape_all = torch.tensor(images_all.shape[-2:])[None].repeat(b*v, 1) # [BxV 2] 

        feat, pose, _ = self._encode_image(images_all, shape_all, intrinsic_embedding_all)
        
        feat = rearrange(feat, "(b v) l c -> b v l c", b=b, v=v)
        pose = rearrange(pose, "(b v) l c -> b v l c", b=b, v=v)

        encoded_feats = torch.split(feat, b, dim=0)
        encoded_pose = torch.split(pose, b, dim=0)

        # TODO: positional embedding
        shapes = rearrange(shape_all, "(b v) c -> b v c", b=b, v=v)
        different_resolution_across_views = not all(torch.equal(shapes[0], shape) for shape in shapes)

        # Initialize an empty list to collect image IDs for each patch.
        # Note that at inference time, different views may have different number of patches.
        image_ids = []

        # Loop through each encoded feature to get the actual number of patches
        for i, encoded_feat in enumerate(encoded_feats):
            num_patches = encoded_feat.shape[1]  # Get the number of patches for this image
            # Extend the image_ids list with the current image ID repeated num_patches times
            image_ids.extend([i] * num_patches)

        image_ids = torch.tensor(image_ids * b).reshape(b, -1).to(encoded_feats[0].device)

        # step 2: decoder
        dec_feat = self._decoder(encoded_feats, encoded_pose, image_ids)
        shape = rearrange(shape_all, "(b v) c -> b v c", b=b, v=v)
        images = rearrange(images_all, "(b v) c h w -> b v c h w", b=b, v=v)

        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'token':
            dec_feat = list(dec_feat)
            for i in range(len(dec_feat)):
                dec_feat[i] = dec_feat[i][:, :, :-1]

        return dec_feat, shape, images

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024
