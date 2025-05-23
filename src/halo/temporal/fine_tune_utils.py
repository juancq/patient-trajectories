from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers.modeling_outputs import SequenceClassifierOutput

from metrics import calculate_metrics

@dataclass
class CustomConfig:
    use_return_dict = False
    def get(self, key, default=None):
        return getattr(self, key, default)

class FineTunedHALO(nn.Module):
    def __init__(self, base_model, hidden_size):
        super(FineTunedHALO, self).__init__()
        #self.config = type('Config', (), {'hidden_size': hidden_size})()
        self.transformer = base_model.transformer
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 1)
        self.config = CustomConfig()

    def get_last_masked_embedding(self, embeddings, mask):
        last_positions = torch.cumsum(mask, dim=1) * mask
        # shape: (batch_size, )
        last_positions = (last_positions != 0).sum(dim=1).long()
        last_positions = last_positions.squeeze()

        batch_size = embeddings.size(0)
        batch_indices = torch.arange(batch_size, device=embeddings.device, dtype=torch.long)

        last_hidden_states = embeddings[batch_indices, last_positions]
        return last_hidden_states

    def gpt_embed(self, embeddings, mask):
        last_positions = torch.cumsum(mask, dim=1) * mask
        # shape: (batch_size, )
        last_positions = ((last_positions !=0).sum(dim=1)-1).long()
        print('last pos', last_positions.shape, last_positions.squeeze().shape)
        #import code
        #code.interact(local=locals())
        last_positions_expanded = last_positions.unsqueeze(1).expand(-1, embeddings.size(2))


        last_positions_expanded = torch.gather(embeddings, 1, last_positions_expanded).squeeze(1)
        last_hidden_states = embeddings[torch.arange(embeddings.size(0)).long(), last_positions]

        #batch_size = embeddings.size(0)
        #batch_indices = torch.arange(batch_size, device=embeddings.device, dtype=torch.long)

        #last_hidden_states = embeddings[batch_indices, last_positions.squeeze(),:]

        #indices = torch.stack([batch_indices, last_positions], dim=1)

        #last_hidden_states = embeddings[batch_indices, last_positions]
        #last_hidden_states = embeddings[indices[:,0], indices[:,1]]
        return last_hidden_states

    def claude_get_last_masked_embedding(self, embeddings, mask):
        last_positions = torch.max((mask*torch.arange(mask.size(1)).to(mask.device)), dim=1)[0]
        last_positions = last_positions.long()

        batch_size = embeddings.size(0)
        batch_indices = torch.arange(batch_size, device=embeddings.device, dtype=torch.long)

        indices = torch.stack([batch_indices, last_positions], dim=1)

        #last_hidden_states = embeddings[batch_indices, last_positions]
        last_hidden_states = embeddings[indices[:,0], indices[:,1]]
        return last_hidden_states

    def forward(self, input_ids, attention_mask=None, labels=None, *args, **kwargs):
        position_ids = None
        hidden_states = self.transformer(input_ids, position_ids)
        # this would have to be changed to use the mask in order to use
        # either the last token (identified via the mask) or to use
        # average of only the non-mask tokens
        #pooled_output = torch.mean(hidden_states, dim=1)
        #pooled_output = self.dropout(pooled_output)
        pooled_output = self.get_last_masked_embedding(hidden_states, attention_mask)
        logits = self.classifier(pooled_output)
        logits = logits.squeeze(1)

        if labels is not None:
            #loss = F.cross_entropy(logits, labels.long())
            loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
            return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states)

        return SequenceClassifierOutput(logits=logits, hidden_states=hidden_states)


def setup_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],#, "c_fc"],
        #target_modules=["c_attn", "c_proj", "c_fc"],
        #target_modules=["transformer.h.5.attn.c_attn", "transformer.h.5.attn.c_proj"],
        modules_to_save=["classifier"],
        lora_dropout=0.1
    )
    peft_model = get_peft_model(model, lora_config)
    print(peft_model.print_trainable_parameters())
    return peft_model