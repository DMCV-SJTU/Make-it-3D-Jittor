import jittor as jt
from jittor import init
from jittor import nn
import clip
from jittor import transform as T


class CLIP(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device  # device
        (self.clip_model, self.clip_preprocess) = clip.load('ViT-B/16', device=self.device, jit=False)
        self.aug = T.Compose([T.Resize((224, 224)), T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    def get_text_embeds(self, prompt, negative_prompt):
        text = clip.tokenize(prompt)# .to(self.device)
        text_z = self.clip_model.encode_text(text)
        text_z = (text_z / text_z.normalize(dim=(- 1)))
        return text_z

    def train_step(self, text_z, pred_rgb):
        pred_rgb = self.aug(pred_rgb)
        image_z = self.clip_model.encode_image(pred_rgb)
        image_z = (image_z / image_z.normalize(dim=(- 1)))
        loss = (- (image_z * text_z).sum((- 1)).mean())
        return loss
