import json
import os
import torch.nn.functional as F
import h5py
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import einops
import open_clip

class PositionalEncodings(nn.Module):
    """Attention is All You Need positional encoding layer"""

    def __init__(self, seq_len, d_model, p_dropout):
        """Initializes the layer."""
        super(PositionalEncodings, self).__init__()
        token_positions = torch.arange(start=0, end=seq_len).view(-1, 1)
        dim_positions = torch.arange(start=0, end=d_model).view(1, -1)
        angles = token_positions / (10000 ** ((2 * dim_positions) / d_model))

        encodings = torch.zeros(1, seq_len, d_model)
        encodings[0, :, ::2] = torch.cos(angles[:, ::2])
        encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
        encodings.requires_grad = False
        self.register_buffer("positional_encodings", encodings)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Performs forward pass of the module."""
        x = x + self.positional_encodings
        x = self.dropout(x)
        return x


class RetrievalModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.encoder_pos_embedding = PositionalEncodings(21, hidden_dim, 0.1)
        encoder_layer_box = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True, dim_feedforward=hidden_dim)
        self.transformer_encoder_box = nn.TransformerEncoder(encoder_layer_box, num_layers=2)
        self.bottleneck_box = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.bottleneck_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
        )
        

    def forward(self, boxes, images):
        boxes = boxes[:, 0:20, :]
        boxes = torch.cat((boxes, images.unsqueeze(dim=1)), dim=1)
        boxes = self.bottleneck_box(boxes)
        encoder_input = self.encoder_pos_embedding(boxes)
        
        encoder_output = self.transformer_encoder_box(encoder_input)
        return encoder_output[:, :20, :], encoder_output[:, -1, :]



####### fill this arguments ###########################
retreival_model_path = 'ComAlign-CLIP-VIT-B32.pt' ### image model path
text_model_path = ''      ### text model path
device = 'cuda'           ### device
train_text_model = False  ### if text model had been trained 
images_path = 'val2017'          ### image directory path
noun_box_images_path = 'embeddings/val2017_noun_yolo_emb_ViT-B-32_openai.h5'
relation_box_images_path = 'embeddings/val2017_rel_yolo_emb_ViT-B-32_openai.h5'
annotations_path = 'captions_val2017.json'  ### caption json file path
noun_sentence_path = 'embeddings/val2017_att_sentence_ViT-B-32_openai_sep.h5'
relation_sentence_path = 'embeddings/val2017_relation_sentence_ViT-B-32_openai_sep.h5'
### model specification from openclip
model_id = 'ViT-B-32'
pretrained = 'openai'
########################################################


clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(model_id, pretrained=pretrained, device=device)
tokenizer = open_clip.get_tokenizer(model_id)

class CocoDetection(Dataset):

    def __init__(self, image_path, image_path_noun, image_path_relation, ann_path, text_path_noun, text_path_relation, image_transform=None):

        self.image_path = image_path
        self.image_path_noun = image_path_noun
        self.image_path_relation = image_path_relation
        self.text_path_noun = text_path_noun
        self.text_path_relation = text_path_relation
        self.ann_path = ann_path
        self.image_ids = []
        self.captions = {}
        self.coco_id_to_index = {}
        self.image_transform = image_transform
        self.count = 0
        
        for annotation in json.load(open(ann_path))['annotations']:
          id = annotation['image_id']
          caption = annotation['caption']
          #######################################
          if id not in self.image_ids:            
              self.image_ids.append(id)
          #######################################
          if id in self.captions:
            self.captions[id].append(caption)
          else:
            self.captions[id] = [caption]

    def get_pil_image(self, index):
        img_id = self.image_ids[index]
        img_name = f"{img_id:012d}.jpg"
        img = Image.open(os.path.join(self.image_path, img_name))
        return img

    def __getitem__(self, index):
        image_noun = h5py.File(self.image_path_noun, 'r')
        image_relation = h5py.File(self.image_path_relation, 'r')
        
        text_noun = h5py.File(self.text_path_noun, 'r')
        text_relation = h5py.File(self.text_path_relation, 'r')
        id = self.image_ids[index]
        try:
            image_noun_encoding = torch.from_numpy(np.array(image_noun.get(str(id)))).to('cpu')
            image_relation_encoding = torch.from_numpy(np.array(image_relation.get(str(id)))).to('cpu')
        except:
            image_noun_encoding = torch.zeros((10, 512))
            image_relation_encoding = torch.zeros((10, 3, 512))
        
        text_noun_encoding = torch.from_numpy(np.array(text_noun.get(str(id)))).to('cpu').squeeze(dim=1)
        text_relation_encoding = torch.from_numpy(np.array(text_relation.get(str(id)))).to('cpu').squeeze(dim=1)
        if text_noun_encoding.shape[0] > 5:
            text_noun_encoding = text_noun_encoding[:5, :, :]
        if text_relation_encoding.shape[0] > 5:
            text_relation_encoding = text_relation_encoding[:5, :, :]
        
        image_noun.close()
        image_relation.close()
        img = self.get_pil_image(index)
        targets = self.captions[id][:5]

        if self.image_transform is not None:
            img_transformed = self.image_transform(img).unsqueeze(0).reshape((3,224,224)).to('cpu')   
                
        return img_transformed, image_noun_encoding, image_relation_encoding, text_noun_encoding, text_relation_encoding, id, targets

    def get_index_from_coco_id(self, img_id):
        return self.coco_id_to_index[img_id]
        
    def __len__(self):
        return len(self.image_ids)


coco_val = CocoDetection(images_path, noun_box_images_path, relation_box_images_path, annotations_path, noun_sentence_path, relation_sentence_path , image_transform=clip_preprocess)


retreival_model = RetrievalModel(hidden_dim=512).to(device)
retreival_model.load_state_dict(torch.load(retreival_model_path))

if train_text_model:
    text_model = RetrievalModel(hidden_dim=512).to(device)
    text_model.load_state_dict(torch.load(text_model_path))

val_loader = DataLoader(dataset=coco_val, batch_size=100, shuffle=False)
retreival_model.eval()

def get_similarity_matrix_single_caption(images, texts, batch_size=5000):
    sim_image2text = torch.tensor([])
    sim_text2image = torch.tensor([])
    
    for i in tqdm(range(0, texts.shape[0], batch_size)):
        cross_sim = einops.einsum(
            images[:, : , :],
            texts[i:i+batch_size, : , :],
            "image_batch patch d , text_batch token d -> image_batch text_batch patch token",
        )
        
        sim_image, _ = cross_sim.max(dim=-1, keepdim=False)
        sim_image = sim_image.mean(dim=-1, keepdim=False)

        sim_text, _ = cross_sim.max(dim=-2, keepdim=False)
        sim_text = sim_text.mean(dim=-1, keepdim=False).T

        sim_image2text = torch.cat((sim_image2text, sim_image), dim=1)
        sim_text2image = torch.cat((sim_text2image, sim_text), dim=0)

    return sim_image2text, sim_text2image

def encode_dataset(clipModel, batch_size=100):
    with torch.no_grad():
        text_to_image_map = []
        image_to_text_map = []
        all_images_noun = []
        all_images_relation = []
        all_texts_noun = []
        all_texts_relation = []
        image_encodings = []
        text_encodings = []
        text_index = 0
        image_index = 0
        all_images_ids = []
        images_cls = []
        texts_cls = []

        for _, (clip_inputs, images_noun, images_relation, texts_noun, texts_relation, id, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
                with torch.no_grad():
                    image_encoding = clipModel.encode_image(clip_inputs.to('cuda')).float()
                    box_inputs = torch.cat((images_noun, images_relation[:, :]), 1).to('cuda')
                    encoded_boxes, image_cls = retreival_model(box_inputs.to('cuda'), image_encoding)
                    images_noun = encoded_boxes[:, :10]
                    images_relation = encoded_boxes[:, 10:]
                        
                targets2 = []
                for i in range(len(targets[0])):
                    targets2.append([targets[j][i] for j in range(len(targets))])
                targetsTokenized = []
                for target in targets2:
                    targetsTokenized.append(tokenizer(target))

                text = torch.stack(targetsTokenized).to(dtype=torch.float32)
                batch_size, captions_per_image, _ = text.shape
                text = torch.flatten(text, start_dim=0, end_dim=1)
                
                text = text.to(torch.long)
                clipModelCuda = clipModel.to('cuda')
                text_encoding = clipModelCuda.encode_text(text.to('cuda'))
                captions = text_encoding.reshape(batch_size, captions_per_image, -1)
                
                if train_text_model:
                    for j in range(texts_noun.shape[1]):
                        text_object_inputs = torch.cat((texts_noun[:, j, :, :], texts_relation[:, j, :, 0, :]), 1).to('cuda')
                        encoded_text_objects, text_cls = text_retrieval(text_object_inputs.to('cuda'), captions[:, j])
                        texts_noun[:, j, :, :] = encoded_text_objects[:, :texts_noun.shape[2]]
                        texts_relation[:, j, :, 0, :] = encoded_text_objects[:, texts_noun.shape[2]:]
                        texts_cls.append(text_cls.cpu())

                for i in range(batch_size):
                    all_images_noun.append(images_noun[i])
                    all_images_relation.append(images_relation[i])
                    all_images_ids.append(id[i])
                    for j in range(captions_per_image):
                        all_texts_noun.append(texts_noun[i][j])
                        all_texts_relation.append(texts_relation[i][j])

                    text_indices = list(range(text_index, text_index + captions_per_image))
                    image_to_text_map.append(text_indices)
                    text_index += captions_per_image
                    text_to_image_map += [image_index] * captions_per_image
                    image_index += 1
                with torch.no_grad():
                    image_encodings.append(image_encoding.cpu())
                    images_cls.append(image_cls.cpu())
                    text_encodings.append(clipModelCuda.encode_text(text.to('cuda')).cpu())                
               
        text_to_image_map = torch.LongTensor(text_to_image_map)
        image_to_text_map = torch.LongTensor(image_to_text_map)
        image_encodings = torch.cat(image_encodings)
        images_cls = torch.cat(images_cls)
        text_encodings = torch.cat(text_encodings)
        all_images_noun = torch.stack(all_images_noun).to(device='cpu')
        all_images_relation = torch.stack(all_images_relation).to(device='cpu')
        all_images_ids = torch.stack(all_images_ids).to(device='cpu')
        all_texts_noun = torch.stack(all_texts_noun).to(device='cpu')
        all_texts_relation = torch.stack(all_texts_relation).to(device='cpu')

        image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
        images_cls = images_cls / images_cls.norm(dim=-1, keepdim=True)
        text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)
        epsilon = 1e-8
        all_images_noun_norm = torch.norm(all_images_noun, dim=-1, keepdim=True)
        all_images_noun = all_images_noun / (all_images_noun_norm + epsilon)
        all_images_relation_norm = torch.norm(all_images_relation, dim=-1, keepdim=True)
        all_images_relation = all_images_relation / (all_images_relation_norm + epsilon)
        all_texts_noun_norm = torch.norm(all_texts_noun, dim=-1, keepdim=True)
        all_texts_noun = all_texts_noun / (all_texts_noun_norm + epsilon)
        all_texts_relation_norm = torch.norm(all_texts_relation, dim=-1, keepdim=True)
        all_texts_relation = all_texts_relation / (all_texts_relation_norm + epsilon)

        return image_encodings, text_encodings, all_images_noun, all_images_relation, all_texts_noun, all_texts_relation,  text_to_image_map, image_to_text_map, all_images_ids, images_cls, texts_cls
    
image_encodings, text_encodings, all_images_noun, all_images_relation, all_texts_noun, all_texts_relation,  text_to_image_map, image_to_text_map, all_images_ids, images_cls, texts_cls = encode_dataset(clip_model.to('cuda').eval())

coarse_sim_matrix_image_2_text = image_encodings @ text_encodings.T
coarse_sim_matrix_image_cls_2_text = images_cls @ text_encodings.T

sim_noun_image_2_text, sim_noun_text_2_image = get_similarity_matrix_single_caption(all_images_noun, all_texts_noun)
sim_relation_image_2_text, sim_relation_text_2_image = get_similarity_matrix_single_caption(all_images_relation, all_texts_relation)

rel_coef = 1
sim_image_2_text = (rel_coef * sim_noun_image_2_text + sim_relation_image_2_text)
sim_text_2_image = (rel_coef * sim_noun_text_2_image + sim_relation_text_2_image)

captions_per_image = image_to_text_map.shape[1]

mean_i2t_sim = torch.mean(torch.max(sim_image_2_text, dim=-1)[0], dim=-1).abs()
mean_t2i_sim = torch.mean(torch.max(sim_text_2_image, dim=-1)[0], dim=-1).abs()
mean_i2t_coarse_sim = torch.mean(torch.max(coarse_sim_matrix_image_2_text, dim=1)[0], dim=-1).abs()
mean_i2t_our_coarse_sim = torch.mean(torch.max(coarse_sim_matrix_image_cls_2_text, dim=1)[0], dim=-1).abs()

sim_image_2_text = sim_image_2_text / mean_i2t_sim
sim_text_2_image = sim_text_2_image / mean_t2i_sim
coarse_sim_matrix_image_2_text = coarse_sim_matrix_image_2_text / mean_i2t_coarse_sim
coarse_sim_matrix_image_cls_2_text = coarse_sim_matrix_image_cls_2_text / mean_i2t_our_coarse_sim


inds = torch.argsort(10 * coarse_sim_matrix_image_2_text.T + sim_text_2_image + coarse_sim_matrix_image_cls_2_text.T, dim=1, descending=True)
text_to_image_recall = []
for k in [1, 5, 10, 50]:
    topk = inds[:, :k]
    correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

    num_correct = correct.sum().item()
    text_to_image_recall.append(num_correct / text_encodings.shape[0])

inds = torch.argsort(0.06 * coarse_sim_matrix_image_cls_2_text + 0.06 * sim_image_2_text + 2 * coarse_sim_matrix_image_2_text + 0.02 * sim_text_2_image.T, dim=1, descending=True)
image_to_text_recall = []
for k in [1, 5, 10, 50]:
    topk = inds[:, :k] 
    topk = topk
    correct = torch.zeros((image_encodings.shape[0],), dtype=torch.bool)

    for i in range(captions_per_image):
        contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
        correct = torch.logical_or(correct, contains_index)

    num_correct = correct.sum().item()
    image_to_text_recall.append(num_correct / image_encodings.shape[0])

print(f'image to text recall: {image_to_text_recall}')
print(f'text to image recall: {text_to_image_recall}')
