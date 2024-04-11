import torch
import transformers
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union


LABEL_LIST = ['No_Propaganda', 'Attack_on_Reputation', 'Manipulative_Wordding']
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



class Gemma(transformers.GemmaPreTrainedModel):
    def __init__(self):
       # super(XLMRobertaBase, self).__init__()
        config = transformers.GemmaConfig()
        config.problem_type = "multi_label_classification"
        config.num_labels = COUNT_TECHNIQUES
        config.return_dict=False
        
        super().__init__(config)
        self.l1 = transformers.GemmaModel.from_pretrained('google/gemma-2b', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(2048, COUNT_TECHNIQUES)


   # def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
       # output_1, _ = self.l1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.l1(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        hidden_states = self.l2(hidden_states)
        logits = self.l3(hidden_states)

        return logits


class Llama(transformers.LlamaPreTrainedModel):
    def __init__(self):
        config = transformers.LlamaConfig()
        config.problem_type = "multi_label_classification"
        config.num_labels = COUNT_TECHNIQUES
        config.return_dict=False

        super().__init__(config)
        self.l1 = transformers.LlamaModel.from_pretrained('meta-llama/Llama-2-7b-hf', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(2048, COUNT_TECHNIQUES)



    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.l1(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        hidden_states = self.l2(hidden_states)
        logits = self.l3(hidden_states)

        return logits
