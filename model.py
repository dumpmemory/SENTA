from transformers import BertPreTrainedModel, BertModel, AutoConfig, AutoTokenizer
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from utils import AnsweringModelOutput, get_label2id
import numpy as np
import torch
import torch.nn.functional as F


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)  # 竖直一分为二
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Baseline(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # self.l1 = nn.Linear(config.hidden_size, 1)
        self.l2 = nn.Linear(768, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sentiment_id=None
    ):

        # print('\n1-------------------------------------------------1')

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooler_output = outputs['pooler_output']  # batch_size * 768

        logits = self.l2(pooler_output)

        if sentiment_id is None:
            return pooler_output, logits

        Loss = nn.CrossEntropyLoss()
        total_loss = Loss(logits, sentiment_id)

        return AnsweringModelOutput(
            loss=total_loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            hidden_states=pooler_output,
            attentions=outputs.attentions,
            sentiment_id=sentiment_id
        )


# class SentimentClassification_Distill(BertPreTrainedModel):
#     def __init__(self, config, biased_model):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#
#         self.bert = BertModel(config)
#
#         # self.in_drop = nn.Dropout(0.5)
#         # self.input_W_G = nn.Linear(768, 300)
#         # self.GGG = GGG(config)
#         self.l1 = nn.Linear(768, 256)
#
#         self.positive = nn.Linear(768, 256)
#         self.neural = nn.Linear(768, 256)
#         self.negative = nn.Linear(768, 256)
#
#         self.classifer = nn.Linear(768, 3)
#
#         self.classifer1 = nn.Linear(512, 3)
#         self.classifer2 = nn.Linear(512, 3)
#         self.classifer3 = nn.Linear(512, 3)
#
#
#         self.init_weights()
#
#         self.biased = biased_model
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         start_positions=None,
#         end_positions=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         sentiment_id=None
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         last_hidden_state = outputs['last_hidden_state']  # batch_size * max_length * 768
#         pooler_output = outputs['pooler_output']  # batch_size * 768
#         context_embedding = pooler_output
#
#         inputs_ = {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'token_type_ids': token_type_ids,
#             'position_ids': position_ids,
#             'head_mask': head_mask,
#             'inputs_embeds': inputs_embeds,
#             'output_attentions': output_attentions,
#             'output_hidden_states': output_hidden_states,
#             'return_dict': return_dict
#         }
#
#         with torch.no_grad():
#             bias_pooler_output, bias_logits = self.biased(
#                 **inputs_
#             )
#
#         student = self.classifer(pooler_output)
#
#         Loss1 = nn.CrossEntropyLoss()
#         Loss2 = nn.CrossEntropyLoss()
#
#         T = 2
#         alpha = 0.5
#
#         loss1 = Loss1(bias_logits/T, sentiment_id)
#         loss2 = Loss2(student, sentiment_id)
#
#         total_loss = alpha * loss1 + (1 - alpha) * loss2
#
#
#         return AnsweringModelOutput(
#             loss=total_loss,
#             logits=student,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             sentiment_id=sentiment_id
#         )


class SentimentClassification(BertPreTrainedModel):
    def __init__(self, config, biased_feature):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        self.l1 = nn.Linear(768, 256)

        self.positive = nn.Linear(768, 256)
        self.neural = nn.Linear(768, 256)
        self.negative = nn.Linear(768, 256)

        self.classifer = nn.Linear(768, 3)
        self.classifer_adjust = nn.Linear(768*2, 3)

        self.init_weights()

        self.biased = biased_feature

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sentiment_id=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs['last_hidden_state']  # batch_size * max_length * 768
        pooler_output = outputs['pooler_output']  # batch_size * 768
        context_embedding = pooler_output

        classifer_output = self.classifer(context_embedding)  # batch_size * 3
        classifer_output = torch.softmax(classifer_output, dim=-1)  # 12 * 3

        biased_feature = (torch.cat(self.biased, dim=-1)).reshape(3, 768)  # 3 * 768

        adjust_feature = torch.matmul(classifer_output, biased_feature)  # 12 * 768

        # concatenate context_embedding with adjust_feature
        adjust_output = self.classifer_adjust(torch.cat((context_embedding, adjust_feature), dim=-1))

        Loss = nn.CrossEntropyLoss()

        total_loss = Loss(adjust_output, sentiment_id)

        return AnsweringModelOutput(
            loss=total_loss,
            logits=adjust_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            sentiment_id=sentiment_id
        )