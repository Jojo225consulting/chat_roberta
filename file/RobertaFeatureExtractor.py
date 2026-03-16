import torch.nn as nn

class RobertaFeatureExtractor(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.roberta = pretrained_model
        #self.fc1 = model.fc1
        # self.relu = model.relu

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        x = outputs.last_hidden_state[:, 0]
        # x = self.fc1(x)
        # x = self.relu(x)
        return x
