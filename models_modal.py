import torch
import torch.nn as nn
import models_vit

from transformers import BertTokenizer, BertModel,  T5ForConditionalGeneration
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_


class multi_modal(nn.Module):
    def __init__(self, vit_model, bert_model, num_classes):
        super(multi_modal, self).__init__()
        self.vit_model = vit_model  
        self.bert_model = bert_model  
        self.fc_classification = nn.Linear(vit_model.embed_dim + bert_model.config.hidden_size, num_classes)
        self.text_gen = T5ForConditionalGeneration.from_pretrained('t5-small')  
        
        # For saliency map generation task
        # self.saliency_head = SaliencyHead(vit_model)  # implement saliency-specific model part
        
    def forward(self, image_input, text_input):
        vit_features = self.vit(image_input)
        bert_features = self.bert(text_input)
        combined_features = torch.cat([vit_features, bert_features], dim=-1)
        classification_output = self.fc_classification(combined_features)
        explanation_input = "Explain why you think this is glaucoma based on the CT scan."
        explanation_output = self.text_gen.generate(input_ids=explanation_input)
        saliency_map = self.saliency_head(image_input)
        return classification_output, explanation_output, saliency_map



vit_model = models_vit.__dict__['vit_large_patch16'](
    num_classes=2,
    drop_path_rate=0.2,
    global_pool=True,
)

# load RETFound weights
checkpoint = torch.load('RETFound_cfp_weights.pth', map_location='cpu')
checkpoint_model = checkpoint['model']
state_dict = vit_model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

interpolate_pos_embed(vit_model, checkpoint_model)
msg = vit_model.load_state_dict(checkpoint_model, strict=False)
assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
trunc_normal_(vit_model.head.weight, std=2e-5)

bert_model = BertModel.from_pretrained('bert-base-uncased')
model = multi_modal(vit_model, bert_model, num_classes=2)  # Assuming binary classification for glaucoma

# Example inputs (dummy data for illustration)
image = torch.randn(1, 3, 224, 224)  # Single image, 3 channels, 224x224
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = ["Predict this image. If this is glaucoma, highlight the region"]  # Text prompt
text_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Forward pass
prediction, attention_map = model(image, text_inputs['input_ids'], text_inputs['attention_mask'])
