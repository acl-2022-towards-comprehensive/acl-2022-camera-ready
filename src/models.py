import torch.nn as nn
import torch
from transformers import (
    BertModel,
    XLNetModel,
    BigBirdModel,
    BertPreTrainedModel,
    XLNetPreTrainedModel,
    BigBirdPreTrainedModel,

    BertForSequenceClassification,
)
from transformers.modeling_utils import SequenceSummary


class BertForPatentPrediction(BertPreTrainedModel):
    def __init__(self, config, trainer_config):
        super().__init__(config)
        self.trainer_config = trainer_config
        self.num_labels = config.num_labels
        self.app_feat_length = trainer_config["app_feat_length"]
        self.mid_dim = trainer_config["mid_dim"]
        self.bert = BertModel(config)

        # Start - Customized Part
        self.fc1 = nn.Linear(config.hidden_size, self.mid_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.classifier = nn.Linear(
            self.mid_dim + self.app_feat_length, self.num_labels
        )
        # End - Customized Part

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

        self.use_hinge_loss = trainer_config["use_hinge_loss"]
        self.device_num = trainer_config["device_num"]

        self.hinge_idx = [0] #if "hinge_idx" not in trainer_config.keys() else trainer_config["hinge_idx"]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # Start - Customized Part
        app_feats=None,  # Added application features
        # End - Customized Part
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = return_dict

        transformer_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        # Yifei: forget why we wanna use the second layer here. 3/25
        pooled_output = transformer_outputs[1]
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.fc1(pooled_output)
        pooled_output_1 = pooled_output.clone()
        pooled_output_2 = pooled_output.clone()
        # store_rep = pooled_output.clone().detach()

        # Start - Customized Part
        if self.app_feat_length > 0:
            pooled_output_1 = torch.cat((pooled_output_1, app_feats.float()), dim=1)
        logits = self.classifier(pooled_output_1)
        logits = self.softmax(logits)
        # End - Customized Part

        if self.use_hinge_loss:
            app_feats_copy = self.manufacture_constraint(
                app_feats, self.device_num, ix_lst=self.hinge_idx
            )
            # print(app_feats_copy)
            pooled_output_2 = torch.cat(
                (pooled_output_2, app_feats_copy.float()), dim=1
            )
            logits_copy = self.classifier(pooled_output_2)
            logits_copy = self.softmax(logits_copy)
            return (
                logits,
                transformer_outputs[2:],
                logits_copy,
                transformer_outputs[2:],
                pooled_output,
            )

        return (logits, transformer_outputs[2:], pooled_output)

        # testing_loss = False # Set to False if you don't want everything being messed up :/
        # loss = None
        # if labels is not None and testing_loss: # only go into this block if you are testing loss function!!
        #     if self.num_labels == 1:
        #         #  We are doing regression
        #         loss_fct = MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #
        # if not return_dict:
        #     output = (logits,) + transformer_outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

    def manufacture_constraint(self, vec, device_num, constraint_lst=[0.5], ix_lst=[0]):
        # First Column is the similarity
        # TODO: similarity_product, max_score_y, max_score_x
        # output multiple monotonic constraint
        # print(len(constraint_lst), len(ix_lst))
        assert len(constraint_lst) == len(ix_lst), "Input Length Mismatch"

        factor_tensor = torch.zeros_like(vec)
        factor_tensor = factor_tensor.fill_(1.0)
        for ix in ix_lst:
            index = torch.tensor([ix]).cuda(device_num)
            # Use the first argument to specify column or raw; Column = 1
            # print("--> Debug:  %s, %s" %(index, constraint_lst[ix_lst.index(ix)]))
            # print(factor_tensor)
            # -> Debug:  tensor([0], device='cuda:1'), 0.5
            factor_tensor = factor_tensor.index_fill_(
                1, index, constraint_lst[ix_lst.index(ix)]
            )

        out = torch.mul(factor_tensor, vec).cuda(device_num)
        return out


# class XLNetForPatentPrediction(XLNetPreTrainedModel):
#     def __init__(self, config, trainer_config):
#         super().__init__(config)
#         # print('config:', config)
#         # print('other config', trainer_config)
#         self.num_labels = config.num_labels
#         self.app_feat_length = trainer_config["app_feat_length"]
#         self.trainer_config = trainer_config
#
#         self.mid_dim = trainer_config["mid_dim"]
#
#         self.transformer = XLNetModel(config)
#         self.sequence_summary = SequenceSummary(config)
#         # config.hidden_dropout_prob = 0.1 # by default
#         # self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#         # Start - Customized Part
#         self.fc1 = nn.Linear(
#             config.d_model, self.mid_dim
#         )  # config.d_model is the default size of xlnet
#         self.softmax = nn.LogSoftmax(dim=1)
#         self.fc2 = nn.Linear(self.mid_dim + self.app_feat_length, self.num_labels)
#         # End - Customized Part
#
#         self.init_weights()
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         # Start - Customized Part
#         app_feats=None,  # Added application features
#         # End - Customized Part
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         # mems=None,
#         # perm_mask=None,
#         # target_mapping=None,
#         # input_mask=None,
#         # use_mems=None,
#     ):
#         return_dict = return_dict
#
#         transformer_outputs = self.transformer(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             # position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             # return_dict=return_dict,
#         )
#
#         pooled_output = transformer_outputs[0]
#         pooled_output = self.sequence_summary(pooled_output)
#         pooled_output = self.fc1(pooled_output)
#
#         # Start - Customized Part
#         # concat app features
#         if self.app_feat_length > 0:
#             pooled_output = torch.cat((pooled_output, app_feats.float()), dim=1)
#         logits = self.fc2(pooled_output)
#         logits = self.softmax(logits)
#         # End - Customized Part
#
#         loss = None
#
#         # if labels is not None:
#         #     if self.num_labels == 1:
#         #         #  We are doing regression
#         #         loss_fct = MSELoss()
#         #         loss = loss_fct(logits.view(-1), labels.view(-1))
#         #     else:
#         #         loss_fct = CrossEntropyLoss()
#         #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#
#         if not return_dict:
#             output = (logits,) + transformer_outputs[1:]
#             return ((loss,) + output) if loss is not None else output
#
#
# class GraphForPatentPrediction:
#     def __init__(self, config):
#         pass


# Class for BigBird
# class BigBirdForPatentPrediction(BigBirdPreTrainedModel):
#     def __init__(self, config, trainer_config):
#         super().__init__(config)
#         self.trainer_config = trainer_config
#         self.num_labels = config.num_labels
#         self.app_feat_length = trainer_config["app_feat_length"]
#         self.mid_dim = trainer_config["mid_dim"]
#         self.bert = BigBirdModel(config)
#
#         # Start - Customized Part
#         self.fc1 = nn.Linear(config.hidden_size, self.mid_dim)
#         self.softmax = nn.LogSoftmax(dim=1)
#         self.classifier = nn.Linear(
#             self.mid_dim + self.app_feat_length, self.num_labels
#         )
#         # End - Customized Part
#
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.init_weights()
#
#         self.use_hinge_loss = trainer_config["use_hinge_loss"]
#         self.device_num = trainer_config["device_num"]
#
#         self.hinge_idx = [0] if "hinge_idx" not in trainer_config.keys() else trainer_config["hinge_idx"]
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         # Start - Customized Part
#         app_feats=None,  # Added application features
#         # End - Customized Part
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the sequence classification/regression loss.
#             Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
#             If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         return_dict = return_dict
#
#         transformer_outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             # return_dict=return_dict,
#         )
#         # Yifei: forget why we wanna use the second layer here. 3/25
#         pooled_output = transformer_outputs[1]
#         pooled_output = self.dropout(pooled_output)
#         pooled_output = self.fc1(pooled_output)
#         pooled_output_1 = pooled_output.clone()
#         pooled_output_2 = pooled_output.clone()
#         # store_rep = pooled_output.clone().detach()
#
#         # Start - Customized Part
#         if self.app_feat_length > 0:
#
#             pooled_output_1 = torch.cat((pooled_output_1, app_feats.float()), dim=1)
#         logits = self.classifier(pooled_output_1)
#         logits = self.softmax(logits)
#         # End - Customized Part
#
#         if self.use_hinge_loss:
#             app_feats_copy = self.manufacture_constraint(
#                 app_feats, self.device_num, ix_lst=self.hinge_idx
#             )
#             # print(app_feats_copy)
#             pooled_output_2 = torch.cat(
#                 (pooled_output_2, app_feats_copy.float()), dim=1
#             )
#             logits_copy = self.classifier(pooled_output_2)
#             logits_copy = self.softmax(logits_copy)
#             return (
#                 logits,
#                 transformer_outputs[2:],
#                 logits_copy,
#                 transformer_outputs[2:],
#                 pooled_output,
#             )
#
#         return (logits, transformer_outputs[2:], pooled_output)
#
#     def manufacture_constraint(self, vec, device_num, constraint_lst=[0.5], ix_lst=[0]):
#         # First Column is the similarity
#         # TODO: similarity_product, max_score_y, max_score_x
#         # output multiple monotonic constraint
#         assert len(constraint_lst) == len(ix_lst), "Input Length Mismatch"
#
#         factor_tensor = torch.zeros_like(vec)
#         factor_tensor = factor_tensor.fill_(1.0)
#         for ix in ix_lst:
#             index = torch.tensor([ix]).cuda(device_num)
#             # Use the first argument to specify column or raw; Column = 1
#             factor_tensor = factor_tensor.index_fill_(
#                 1, index, constraint_lst[ix_lst.index(ix)]
#             )
#
#         out = torch.mul(factor_tensor, vec).cuda(device_num)
#         return out
#
# class Bert2LayerForPatentPrediction(BertPreTrainedModel):
#     def __init__(self, config, trainer_config):
#         super().__init__(config)
#         self.trainer_config = trainer_config
#         self.num_labels = config.num_labels
#         self.app_feat_length = trainer_config["app_feat_length"]
#         self.mid_dim = trainer_config["mid_dim"]
#         self.bert = BertModel(config)
#
#         # Start - Customized Part
#         self.fc1 = nn.Linear(config.hidden_size, self.mid_dim) #unused
#         self.softmax = nn.LogSoftmax(dim=1)
#         self.classifier = nn.Sequential(
#             nn.Linear(config.hidden_size + self.app_feat_length, self.mid_dim),
#             nn.ReLU(),
#             nn.Linear(self.mid_dim,  self.num_labels)
#         )
#         # nn.Linear(
#         #     self.mid_dim + self.app_feat_length, self.num_labels
#         # )
#
#         # End - Customized Part
#
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.init_weights()
#
#         self.use_hinge_loss = trainer_config["use_hinge_loss"]
#         self.device_num = trainer_config["device_num"]
#         self.hinge_idx = [trainer_config["hinge_idx"]]
#         try:
#             self.use_epsilon = trainer_config["use_epsilon"]
#             self.hinge_epsilon = trainer_config['hinge_epsilon']
#         except:
#             self.use_epsilon = False
#             self.hinge_epsilon = None
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         # Start - Customized Part
#         app_feats=None,  # Added application features
#         # End - Customized Part
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the sequence classification/regression loss.
#             Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
#             If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         return_dict = return_dict
#
#         transformer_outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             # return_dict=return_dict,
#         )
#
#         pooled_output = transformer_outputs[1]
#         pooled_output = self.dropout(pooled_output)
#         # pooled_output = self.fc1(pooled_output)
#         pooled_output_1 = pooled_output.clone()
#         pooled_output_2 = pooled_output.clone()
#         # store_rep = pooled_output.clone().detach()
#
#         # Start - Customized Part
#         if self.app_feat_length > 0:
#             pooled_output_1 = torch.cat((pooled_output_1, app_feats.float()), dim=1)
#         logits = self.classifier(pooled_output_1)
#         logits = self.softmax(logits)
#         # End - Customized Part
#
#         if self.use_hinge_loss:
#             app_feats_copy = self.manufacture_constraint(
#                 app_feats, self.use_epsilon, self.hinge_epsilon, self.device_num, ix_lst=self.hinge_idx
#             )
#             # print(app_feats_copy)
#             pooled_output_2 = torch.cat(
#                 (pooled_output_2, app_feats_copy.float()), dim=1
#             )
#             logits_copy = self.classifier(pooled_output_2)
#             logits_copy = self.softmax(logits_copy)
#             return (
#                 logits,
#                 transformer_outputs[2:],
#                 logits_copy,
#                 transformer_outputs[2:],
#                 pooled_output,
#             )
#
#         return (logits, transformer_outputs[2:], pooled_output)
#
#
#     def manufacture_constraint(self, vec, use_epsilon, hinge_epsilon, device_num, constraint_lst=[0.5], ix_lst=[0]):
#         # First Column is the similarity
#         # TODO: similarity_product, max_score_y, max_score_x
#         # output multiple monotonic constraint
#         if use_epsilon:
#             out = torch.clone(vec)
#             for ix in ix_lst:
#                 out[:,ix] -= hinge_epsilon
#         else:
#             assert len(constraint_lst) == len(ix_lst), "Input Length Mismatch"
#
#             factor_tensor = torch.zeros_like(vec)
#             factor_tensor = factor_tensor.fill_(1.0)
#             for ix in ix_lst:
#                 index = torch.tensor([ix]).cuda(device_num)
#                 # Use the first argument to specify column or raw; Column = 1
#                 factor_tensor = factor_tensor.index_fill_(
#                     1, index, constraint_lst[ix_lst.index(ix)]
#                 )
#
#             out = torch.mul(factor_tensor, vec).cuda(device_num)
#         return out
#
# class Bert3LayerForPatentPrediction(BertPreTrainedModel):
#     def __init__(self, config, trainer_config):
#         super().__init__(config)
#         self.trainer_config = trainer_config
#         self.num_labels = config.num_labels
#         self.app_feat_length = trainer_config["app_feat_length"]
#         self.mid_dim = trainer_config["mid_dim"]
#         self.bert = BertModel(config)
#
#         # Start - Customized Part
#         self.fc1 = nn.Linear(config.hidden_size, self.mid_dim)
#         self.softmax = nn.LogSoftmax(dim=1)
#         self.classifier = nn.Sequential(
#             nn.Linear(self.mid_dim + self.app_feat_length, 512),
#             nn.ReLU(),
#             nn.Linear(512, self.num_labels)
#         )
#         # nn.Linear(
#         #     self.mid_dim + self.app_feat_length, self.num_labels
#         # )
#
#         # End - Customized Part
#
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.init_weights()
#
#         self.use_hinge_loss = trainer_config["use_hinge_loss"]
#         self.device_num = trainer_config["device_num"]
#         self.hinge_idx = [trainer_config["hinge_idx"]]
#         try:
#             self.use_epsilon = trainer_config["use_epsilon"]
#             self.hinge_epsilon = trainer_config['hinge_epsilon']
#         except:
#             self.use_epsilon = False
#             self.hinge_epsilon = None
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         # Start - Customized Part
#         app_feats=None,  # Added application features
#         # End - Customized Part
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the sequence classification/regression loss.
#             Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
#             If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         return_dict = return_dict
#
#         transformer_outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             # return_dict=return_dict,
#         )
#
#         pooled_output = transformer_outputs[1]
#         pooled_output = self.dropout(pooled_output)
#         pooled_output = self.fc1(pooled_output)
#         pooled_output_1 = pooled_output.clone()
#         pooled_output_2 = pooled_output.clone()
#         # store_rep = pooled_output.clone().detach()
#
#         # Start - Customized Part
#         if self.app_feat_length > 0:
#             pooled_output_1 = torch.cat((pooled_output_1, app_feats.float()), dim=1)
#         logits = self.classifier(pooled_output_1)
#         logits = self.softmax(logits)
#         # End - Customized Part
#
#         if self.use_hinge_loss:
#             app_feats_copy = self.manufacture_constraint(
#                 app_feats, self.use_epsilon, self.hinge_epsilon, self.device_num, ix_lst=self.hinge_idx
#             )
#             # print(app_feats_copy)
#             pooled_output_2 = torch.cat(
#                 (pooled_output_2, app_feats_copy.float()), dim=1
#             )
#             logits_copy = self.classifier(pooled_output_2)
#             logits_copy = self.softmax(logits_copy)
#             return (
#                 logits,
#                 transformer_outputs[2:],
#                 logits_copy,
#                 transformer_outputs[2:],
#                 pooled_output,
#             )
#
#         return (logits, transformer_outputs[2:], pooled_output)
#
#
#     def manufacture_constraint(self, vec, use_epsilon, hinge_epsilon, device_num, constraint_lst=[0.5], ix_lst=[0]):
#         # First Column is the similarity
#         # TODO: similarity_product, max_score_y, max_score_x
#         # output multiple monotonic constraint
#         if use_epsilon:
#             out = torch.clone(vec)
#             for ix in ix_lst:
#                 out[:,ix] -= hinge_epsilon
#         else:
#             assert len(constraint_lst) == len(ix_lst), "Input Length Mismatch"
#
#             factor_tensor = torch.zeros_like(vec)
#             factor_tensor = factor_tensor.fill_(1.0)
#             for ix in ix_lst:
#                 index = torch.tensor([ix]).cuda(device_num)
#                 # Use the first argument to specify column or raw; Column = 1
#                 factor_tensor = factor_tensor.index_fill_(
#                     1, index, constraint_lst[ix_lst.index(ix)]
#                 )
#
#             out = torch.mul(factor_tensor, vec).cuda(device_num)
#         return out
#
# class Bert4LayerForPatentPrediction(BertPreTrainedModel):
#     def __init__(self, config, trainer_config):
#         super().__init__(config)
#         self.trainer_config = trainer_config
#         self.num_labels = config.num_labels
#         self.app_feat_length = trainer_config["app_feat_length"]
#         self.mid_dim = trainer_config["mid_dim"]
#         self.bert = BertModel(config)
#
#         # Start - Customized Part
#         self.fc1 = nn.Linear(config.hidden_size, self.mid_dim)
#         self.softmax = nn.LogSoftmax(dim=1)
#         self.classifier = nn.Sequential(
#             nn.Linear(self.mid_dim + self.app_feat_length, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, self.num_labels)
#         )
#         # nn.Linear(
#         #     self.mid_dim + self.app_feat_length, self.num_labels
#         # )
#
#         # End - Customized Part
#
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.init_weights()
#
#         self.use_hinge_loss = trainer_config["use_hinge_loss"]
#         self.device_num = trainer_config["device_num"]
#         self.hinge_idx = [trainer_config["hinge_idx"]]
#         try:
#             self.use_epsilon = trainer_config["use_epsilon"]
#             self.hinge_epsilon = trainer_config['hinge_epsilon']
#         except:
#             self.use_epsilon = False
#             self.hinge_epsilon = None
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         # Start - Customized Part
#         app_feats=None,  # Added application features
#         # End - Customized Part
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the sequence classification/regression loss.
#             Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
#             If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         return_dict = return_dict
#
#         transformer_outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             # return_dict=return_dict,
#         )
#
#         pooled_output = transformer_outputs[1]
#         pooled_output = self.dropout(pooled_output)
#         pooled_output = self.fc1(pooled_output)
#         pooled_output_1 = pooled_output.clone()
#         pooled_output_2 = pooled_output.clone()
#         # store_rep = pooled_output.clone().detach()
#
#         # Start - Customized Part
#         if self.app_feat_length > 0:
#             pooled_output_1 = torch.cat((pooled_output_1, app_feats.float()), dim=1)
#         logits = self.classifier(pooled_output_1)
#         logits = self.softmax(logits)
#         # End - Customized Part
#
#         if self.use_hinge_loss:
#             app_feats_copy = self.manufacture_constraint(
#                 app_feats, self.use_epsilon, self.hinge_epsilon, self.device_num, ix_lst=self.hinge_idx
#             )
#             # print(app_feats_copy)
#             pooled_output_2 = torch.cat(
#                 (pooled_output_2, app_feats_copy.float()), dim=1
#             )
#             logits_copy = self.classifier(pooled_output_2)
#             logits_copy = self.softmax(logits_copy)
#             return (
#                 logits,
#                 transformer_outputs[2:],
#                 logits_copy,
#                 transformer_outputs[2:],
#                 pooled_output,
#             )
#
#         return (logits, transformer_outputs[2:], pooled_output)
#
#
#     def manufacture_constraint(self, vec, use_epsilon, hinge_epsilon, device_num, constraint_lst=[0.5], ix_lst=[0]):
#         # First Column is the similarity
#         # TODO: similarity_product, max_score_y, max_score_x
#         # output multiple monotonic constraint
#         if use_epsilon:
#             out = torch.clone(vec)
#             for ix in ix_lst:
#                 out[:,ix] -= hinge_epsilon
#         else:
#             assert len(constraint_lst) == len(ix_lst), "Input Length Mismatch"
#
#             factor_tensor = torch.zeros_like(vec)
#             factor_tensor = factor_tensor.fill_(1.0)
#             for ix in ix_lst:
#                 index = torch.tensor([ix]).cuda(device_num)
#                 # Use the first argument to specify column or raw; Column = 1
#                 factor_tensor = factor_tensor.index_fill_(
#                     1, index, constraint_lst[ix_lst.index(ix)]
#                 )
#
#             out = torch.mul(factor_tensor, vec).cuda(device_num)
#         return out
