import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from transformers import BertModel, BertConfig

class BertClassifier(nn.Module):

    def __init__(self, freeze_bert=False):

        super(BertClassifier, self).__init__()

        D_in, H, D_out = 768, 50, 3
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        return logits

class DUMA_Layer(nn.Module):

    def __init__(self, d_model_size, num_heads):

        super(DUMA_Layer, self).__init__()
        self.attn_qa = MultiheadAttention(d_model_size, num_heads)
        self.attn_p = MultiheadAttention(d_model_size, num_heads)

    def forward(self, qa_seq_representation, p_seq_representation, qa_mask=None, p_mask=None):

        qa_seq_representation = qa_seq_representation.permute([1, 0, 2])
        p_seq_representation = p_seq_representation.permute([1, 0, 2])
        enc_output_qa, _ = self.attn_qa(
            value=qa_seq_representation,
            key=qa_seq_representation,
            query=p_seq_representation
        )
        enc_output_p, _ = self.attn_p(
            value=p_seq_representation,
            key=p_seq_representation,
            query=qa_seq_representation
        )

        return enc_output_qa.permute([1, 0, 2]), enc_output_p.permute([1, 0, 2])

class DUMA(nn.Module):

    def __init__(self, freeze_bert=False):
        super(DUMA, self).__init__()

        D_in, H, D_out = 768, 50, 3
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.duma = DUMA_Layer(D_in, num_heads=self.config.num_attention_heads)
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids_p, attention_mask_p, input_ids_qa, attention_mask_qa):

        # input_ids_p, input_ids_qa, attention_mask_p, attention_mask_qa = seperate(input_ids, attention_mask)
        outputs_qa = self.bert(input_ids=input_ids_qa,
                               attention_mask=attention_mask_qa)
        outputs_p = self.bert(input_ids=input_ids_p,
                              attention_mask=attention_mask_p)
        last_outputs_qa = outputs_qa.last_hidden_state
        last_outputs_p = outputs_p.last_hidden_state
        enc_outputs_qa_0, enc_outputs_p_0 = self.duma(last_outputs_qa,
                                                  last_outputs_p,
                                                  attention_mask_qa,
                                                  attention_mask_p)
        # try different number of DUMA Layers, default number = 1
        # enc_outputs_p_1, enc_outputs_qa_1 = self.duma(enc_outputs_qa_0,
        #                                           enc_outputs_p_0,
        #                                           attention_mask_qa,
        #                                           attention_mask_p)
        # enc_outputs_p_2, enc_outputs_qa_2 = self.duma(enc_outputs_qa_1,
        #                                               enc_outputs_p_1,
        #                                               attention_mask_qa,
        #                                               attention_mask_p)
        # enc_outputs_p_3, enc_outputs_qa_3 = self.duma(enc_outputs_qa_2,
        #                                               enc_outputs_p_2,
        #                                               attention_mask_qa,
        #                                               attention_mask_p)
        # enc_outputs_p_4, enc_outputs_qa_4 = self.duma(enc_outputs_qa_3,
        #                                               enc_outputs_p_3,
        #                                               attention_mask_qa,
        #                                               attention_mask_p)
        # enc_outputs_p_5, enc_outputs_qa_5 = self.duma(enc_outputs_qa_4,
        #                                               enc_outputs_p_4,
        #                                               attention_mask_qa,
        #                                               attention_mask_p)
        fuse_output = torch.cat([enc_outputs_qa_0, enc_outputs_p_0], dim=1)

        pooled_output = torch.mean(fuse_output, dim=1)
        logits = self.classifier(pooled_output)

        return logits

# models for the silent films task
class C3D(nn.Module):
    '''
    The C3D network
    '''
    
    def __init__(self,seq_len,feature_dim):  #feature_dim = 768
        super(C3D,self).__init__()
        
        # x = [batch_size,3,32,112,112]
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))    # x = [batch_size,64,32,112,112]
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))         # x = [batch_size,64,32,56,56]

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # x = [batch_size,128,32,56,56]
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))         # x = [batch_size,128,16,28,28]

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))# x = [batch_size,256,16,28,28]
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))# x = [batch_size,256,16,28,28]
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))         # x = [batch_size,256,8,14,14]

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))# x = [batch_size,512,8,14,14]
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))# x = [batch_size,512,8,14,14]
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))         # x = [batch_size,512,4,7,7]

        self.conv5a = nn.Conv3d(512,512, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # x = [batch_size,512,4,7,7]
        self.conv5b = nn.Conv3d(512,512, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # x = [batch_size,512,4,7,7]
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),padding=(0, 1, 1))         # x = [batch_size,512,2,4,4]

        self.conv6 = nn.Conv3d(512,512, kernel_size=(3, 3, 3), padding=(1, 1, 1))   # x = [batch_size,512,2,4,4]
        self.pool6 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))          # x = [batch_size,512,2,2,2]

        self.fc5 = nn.Linear(4096,seq_len*feature_dim)

        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(p=0.5)

        self.relu =  nn.ReLU()
        

    def forward(self,x):  #x = [batch_size,3,32,56,56]
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = self.relu(self.conv6(x))
        x = self.pool6(x)

        x = x.view(-1, 4096)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)

        #Video_feature_Out = self.fc7(x)
        Video_feature_Out = x.view(-1,self.seq_len,self.feature_dim)
        
        return Video_feature_Out   #output_size = [batch_size,feature_dim=768]
    
class V_DUMA_Layer(nn.Module):

    def __init__(self, d_model_size, num_heads):

        super(V_DUMA_Layer, self).__init__()
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
        self.duma = V_DUMA_Layer(D_in, num_heads=self.config.num_attention_heads)
        #self.Linear = nn.Linear(H_v, Feat_dim)
        
        self.C3D = C3D(8, Feat_dim)


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