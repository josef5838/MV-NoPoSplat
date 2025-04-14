from safetensors.torch import load_file
import torch
faster_weights = load_file("/cluster/project/cvg/students/mv-nopo/MV-NoPoSplat/pretrained_weights/Fast3R_ViT_Large_512/model.safetensors")
faster_encoder_keys = [key for key in faster_weights.keys() if key.startswith("encoder")]
faster_encoder_dict = {key: faster_weights[key] for key in faster_encoder_keys}

nopo_weights = torch.load("/cluster/project/cvg/students/mv-nopo/MV-NoPoSplat/pretrained_weights/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
nopo_encoder_keys = [key for key in nopo_weights.keys() if key.startswith("encoder")]
nopo_encoder_dict = {key: nopo_weights[key] for key in nopo_encoder_keys}

import pdb; pdb.set_trace()

# torch.save(encoder_dict, "/cluster/project/cvg/students/mv-nopo/MV-NoPoSplat/pretrained_weights/Fast3R_Encoder.pth")
