import torch

H = W = 28

center_i = H // 2
center_j = W // 2

def offset_to_patch_id(dx,dy):
    i = min(max(0, center_i + dy), H - 1)
    j = min(max(0, center_j + dx), W - 1)
    return i * W + j
a = torch.randn(2, 1, 4)
b = torch.randn(2,3,4)
output = torch.nn.functional.cosine_similarity(a, b, dim=-1)

print(output.shape)