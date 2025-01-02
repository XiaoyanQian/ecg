import torch
import torch.nn as nn
import models.resnet1d as resnet
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer
from models.vit1d import vit_base, vit_small, vit_tiny, vit_middle


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, spacial_dim + 1, embed_dim) / embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(0, 2, 1)  # convert X shape (B, C, L) to (B, L, C)

        self.cls_tokens = self.cls_token + self.positional_embedding[:, :1, :]
        self.cls_tokens = self.cls_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((self.cls_tokens, x), dim=1)
        x = x + self.positional_embedding[:, :, :].to(x.dtype)  # (L+1)NC
        x, att_map = self.mhsa(x[:, :1, :], x, x, average_attn_weights=True)
        x = self.c_proj(x)
        return x.squeeze(0), att_map[:, :, 1:]


class ECGClip(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        projection_head = config['projection_head']
        self.proj_hidden = projection_head['mlp_hidden_size']
        self.proj_out = projection_head['projection_size']
        self.ecg_model = config['ecg_model']
        self.num_leads = config['num_leads']

        if self.ecg_model in resnet.__dict__:
            ecg_encoder = resnet.__dict__[self.ecg_model]()
            self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1)
            self.att_pool_head = AttentionPool2d(
                spacial_dim=313,
                embed_dim=self.proj_out,
                num_heads=4,
                output_dim=self.proj_out)
            self.linear1 = nn.Linear(self.proj_out, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_out, self.proj_out, bias=False)
        elif 'vit' in self.ecg_model:
            if self.ecg_model == 'vit_tiny':
                ecg_encoder = vit_tiny(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_small':
                ecg_encoder = vit_small(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_middle':
                ecg_encoder = vit_middle(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_base':
                ecg_encoder = vit_base(num_leads=self.num_leads)
            else:
                raise ValueError(f'Unknown model: {self.ecg_model}')

            self.proj_emb_width = ecg_encoder.width
            self.proj_e = nn.Sequential(
                nn.Linear(self.proj_emb_width, self.proj_hidden),
                nn.BatchNorm1d(self.proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.proj_hidden, self.proj_out),
                nn.BatchNorm1d(self.proj_out),
            )
            self.linear1 = nn.Linear(self.proj_emb_width, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_emb_width, self.proj_out, bias=False)
        else:
            raise ValueError(f'Unknown model: {self.ecg_model}')

        # ecg
        self.ecg_encoder = ecg_encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        # text encoder
        url = config['text_model']
        self.lm_model = AutoModel.from_pretrained(url, trust_remote_code=True, revision='main')
        self.tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True, revision='main')
        self.proj_t = nn.Sequential(
            nn.Linear(768, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out),
        )  # text projector

    def _tokenize(self, text):
        return self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            add_special_tokens=True,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt')

    @torch.no_grad()
    def ext_ecg_emb(self, ecg):
        embed = self.ecg_encoder(ecg)
        if 'resnet' in self.ecg_model.lower():
            embed = self.downconv(embed)
            proj_emb = self.att_pool_head(embed)[0]
            proj_emb = proj_emb.view(proj_emb.shape[0], -1)
        if 'vit' in self.ecg_model:
            proj_emb = self.proj_e(embed)
        return proj_emb

    @torch.no_grad()
    def get_text_emb(self, input_ids, attention_mask):
        return self.lm_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

    def forward(self, ecg, input_ids, attention_mask):
        # ecg features
        ecg_emb = self.ecg_encoder(ecg)
        if 'resnet' in self.ecg_model:
            # attention pooling (only for resnet models)
            ecg_emb = self.downconv(ecg_emb)
            proj_ecg_emb, _ = self.att_pool_head(ecg_emb)
            proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)
            ecg_emb = self.pool(ecg_emb).view(ecg_emb.shape[0], -1)
            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))
        if 'vit' in self.ecg_model:
            proj_ecg_emb = self.proj_e(ecg_emb)
            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))  # branch 1
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))  # branch 2
        proj_ecg_emb = normalize(proj_ecg_emb, dim=-1)

        # text features
        text_emb = self.get_text_emb(input_ids, attention_mask)
        proj_text_emb = self.proj_t(text_emb.contiguous())
        proj_text_emb = normalize(proj_text_emb, dim=-1)

        return {
            'ecg_emb': [ecg_emb1, ecg_emb2],
            'proj_ecg_emb': [proj_ecg_emb],
            'proj_text_emb': [proj_text_emb]
        }


class MultiModalClassifier(nn.Module):
    def __init__(self, base_model, labels, use_cam_mode=False):
        super().__init__()

        self.model = base_model
        self.labels = labels
        self.use_cam_mode = use_cam_mode

    def forward(self, ecg, text=None):
        if ecg.dim() == 4:
            ecg = ecg.squeeze(2)
    
        if self.use_cam_mode:
            text = [self.labels[0]]

        tokenizer_output = self.model._tokenize(text)
        input_ids = tokenizer_output['input_ids'].to(ecg.device)
        attention_mask = tokenizer_output['attention_mask'].to(ecg.device)
        outputs = self.model(ecg=ecg, input_ids=input_ids, attention_mask=attention_mask)

        proj_ecg_emb = outputs['proj_ecg_emb'][0]
        proj_text_emb = outputs['proj_text_emb'][0]

        logits_per_ecg =  proj_ecg_emb@proj_text_emb.T
        if self.use_cam_mode:
            return logits_per_ecg
        else:
            return logits_per_ecg.softmax(dim=-1)
