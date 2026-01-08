import torch
from pi_zero_pytorch import PiZero

model = PiZero.from_checkpoint()

images = torch.randn(1, 3, 224, 224)
token_ids = torch.randint(0, 257152, (1, 48))

model.eval()
with torch.no_grad():
    logits = model.forward_only_vision_language(images, token_ids)

assert logits.shape == (1, 304, 257152)
