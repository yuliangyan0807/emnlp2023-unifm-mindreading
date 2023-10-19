import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from transformers import BertModel, BertConfig
from c3d import *

class DUMA_Layer(nn.Module):

    def __init__(self, d_model_size, num_heads):

        super(DUMA_Layer, self).__init__()
        self.attn_qr = MultiheadAttention(d_model_size, num_heads)
        self.attn_v = MultiheadAttention(d_model_size, num_heads)

    def forward(self, qr_representation, v_representation):

        qr_representation = qr_representation.permute([1, 0, 2])
        v_representation = v_representation.permute([1, 0, 2])
        enc_output_qr, _ = self.attn_qr(
            value=qr_representation,
            key=qr_representation,
            query=v_representation
        )
        enc_output_v, _ = self.attn_v(
            value=v_representation,
            key=v_representation,
            query=qr_representation
        )

        return enc_output_qr.permute([1, 0, 2]), enc_output_v.permute([1, 0, 2])

class V_DUMA(nn.Module):

    def __init__(self):
        super(V_DUMA, self).__init__()
        D_in, H, H_v, D_out, Feat_dim = 768, 50, 1024, 3, 768
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.duma = DUMA_Layer(D_in, num_heads=self.config.num_attention_heads)
        #self.Linear = nn.Linear(H_v, Feat_dim)
        
        self.C3D = C3D(8,Feat_dim)


        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(H, D_out)
        )

    def forward(self, input_ids_qr, attention_mask_qr, imgs_tensors):
        outputs_qr = self.bert(input_ids=input_ids_qr,
                               attention_mask=attention_mask_qr)
        #(batch_size, seq_length, 768)
        last_outputs_qr = outputs_qr.last_hidden_state


        last_outputs_v = self.C3D(imgs_tensors)          #[batch_size, 8, 768]


        # duma layer
        enc_outputs_qr, enc_outputs_v = self.duma(last_outputs_qr, last_outputs_v)
        # increase the number of duma layers.
        # enc_outputs_qr, enc_outputs_v = self.duma(enc_outputs_qr, enc_outputs_v)
        # enc_outputs_qr, enc_outputs_v = self.duma(enc_outputs_qr, enc_outputs_v)
        # enc_outputs_qr, enc_outputs_v = self.duma(enc_outputs_qr, enc_outputs_v)
        # enc_outputs_qr, enc_outputs_v = self.duma(enc_outputs_qr, enc_outputs_v)
        # enc_outputs_qr, enc_outputs_v = self.duma(enc_outputs_qr, enc_outputs_v)
        fuse_output = torch.cat([enc_outputs_qr, enc_outputs_v], dim=1)

        # fuse_output = torch.cat([last_outputs_qr,last_outputs_v], dim=1)

        pooled_output = torch.mean(fuse_output, dim=1)
        logits = self.classifier(pooled_output)

        return logits