
from safetensors.torch import load_file
import torch

print("loading checkpoints")
ckpt_weights = torch.load("../pretrained_weights/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth", map_location='cpu')
# dusterweights = torch.load("../pretrained_weights/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth", map_location='cpu')
print("load success")

model_dict = ckpt_weights['model']
# args_dict = ckpt_weights['args']

keys = list(model_dict.keys())

with open("keys.txt", "w") as f:
    for key in keys:
        f.write(key + "\n")

fast3r_weights = load_file("../pretrained_weights/Fast3R_ViT_Large_512/model.safetensors")


fast3r_keys = list(fast3r_weights.keys())

with open("fast3r_keys.txt", "w") as f:
    for key in fast3r_keys:
        f.write(key + "\n")



final_out = {}
name ={"downstream_head1.dpt.scratch.layer_rn.0.weight":"downstream_head.dpt.scratch.layer1_rn.weight", 
       "downstream_head1.dpt.scratch.layer_rn.1.weight":"downstream_head.dpt.scratch.layer2_rn.weight",
       "downstream_head1.dpt.scratch.layer_rn.2.weight":"downstream_head.dpt.scratch.layer3_rn.weight",
       "downstream_head1.dpt.scratch.layer_rn.3.weight":"downstream_head.dpt.scratch.layer4_rn.weight",}
name_keys = list(name.keys())

name_2 ={"downstream_head2.dpt.scratch.layer_rn.0.weight":"downstream_head_local.dpt.scratch.layer1_rn.weight", 
       "downstream_head2.dpt.scratch.layer_rn.1.weight":"downstream_head_local.dpt.scratch.layer2_rn.weight",
       "downstream_head2.dpt.scratch.layer_rn.2.weight":"downstream_head_local.dpt.scratch.layer3_rn.weight",
       "downstream_head2.dpt.scratch.layer_rn.3.weight":"downstream_head_local.dpt.scratch.layer4_rn.weight",}
name_keys_2 = list(name_2.keys())

with open("fast3r_keys.txt", "r") as f:
    for key in f:
        key = key.strip()
        if 'encoder.' in key : 
            final_out[key[8:]] = fast3r_weights[key]
        # elif 'decoder.dec_norm' in key:
        #     final_out[key[8:]] = model_dict[key[8:]]
        elif key in name_keys:
            final_out[key] = fast3r_weights[name[key]]
        elif key in name_keys_2:
            final_out[key] = fast3r_weights[name_2[key]]
        elif 'decoder.' in key:
            final_out[key[8:]] = fast3r_weights[key]
        elif 'downstream_head_local' in key:
            final_out["downstream_head2"+key[21:]] = fast3r_weights[key]
        elif 'downstream_head' in key:
            final_out["downstream_head1"+key[15:]] = fast3r_weights[key]
        else:
            final_out[key] = model_dict[key]

final_out_keys = list(final_out.keys())
print(len(final_out_keys))

with open("final_keys.txt", "w") as f:
    for key in final_out_keys:
        f.write(key + "\n")


ckpt_weights['model'] = final_out

torch.save(ckpt_weights, '../pretrained_weights/Fast3R_MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')


