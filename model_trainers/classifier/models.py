import torch
import transformers

LABEL_LIST = ('Attack_on_Reputation', 'Manipulative_Wordding')
COUNT_TECHNIQUES=len(LABEL_LIST)

class XLMRobertaBase(torch.nn.Module):
    def __init__(self):
        super(XLMRobertaBase, self).__init__()
        self.l1 = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base', return_dict=False, add_pooling_layer=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, COUNT_TECHNIQUES)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output_1, _ = self.l1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class XLMRobertaLarge(torch.nn.Module):
    def __init__(self):
        super(XLMRobertaLarge, self).__init__()
        self.l1 = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-large', return_dict=False, add_pooling_layer=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(1024, COUNT_TECHNIQUES)

    def forward(self, ids, mask, token_type_ids=None):
        output_1, _ = self.l1(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
