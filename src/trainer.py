# Author: Zhaoyi Hou (Joey), Yifei Ning (Couson)
# Last Update: 3/18/2022
# Inspired by Huggingface Trainer (see below)
# Reference: https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/trainer.py#L199

import numpy as np
import pandas as pd
import string
import re
import random
from tqdm import tqdm
import time
import datetime
import math
import pickle
# from ast import literal_eval
import os
import sys

# nltk
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk import sent_tokenize
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words("english"))

# torch
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    random_split,
    RandomSampler,
    SequentialSampler,
)

# Pre-processing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight

# other models
from sklearn.linear_model import LogisticRegression  # logistic regression

# metrics
from sklearn import metrics  # accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# transformer
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BigBirdTokenizer
from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification

# from transformers.modeling_bert import BertPreTrainedModel
from transformers import get_linear_schedule_with_warmup

# src/
from preprocessing import *
from models import BertForPatentPrediction
from reg import *

######## Trainer Class #########
class Trainer:
    def __init__(self, trainer_config):

        self.trainer_config = trainer_config

        # Check if this is just a test
        if trainer_config["testing"] == 1:
            print("==> [Trainer.py: __init__(self, trainer_config)] It's just testing!")
            self.raw_data_path = "./data/test/"
            self.testing = True
        else:
            self.raw_data_path = "/data4/xuanyu/full_data_2021/raw/"
            self.testing = False

        # In checkpoints/ folder, there will be a folder with the same name as the model
        self.checkpoint_dir = (
            "/data1/xuanyu/checkpoints/" + trainer_config["model_name"] + "/"
        )
        os.system("mkdir -p /data1/xuanyu/checkpoints/" + trainer_config["model_name"])
        self.from_checkpoint = int(trainer_config["from_checkpoint"])

        self.dataloaders_folder = "/data4/xuanyu/full_data_2021/dataloaders/"
        self.dataloader_name = trainer_config["dataloader_name"]
        self.dataloaders_dir = self.dataloaders_folder + self.dataloader_name + "/"

        # exploded status
        if trainer_config["exploded"] == 1:
            self.exploded = True
        else:
            self.exploded = False

        # Checking GPU
        self.device_num = int(trainer_config["device_num"])
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(self.device_num))
            print(
                "==> [Trainer.py: __init__(self, trainer_config)] Current device:",
                self.device,
            )
        torch.cuda.empty_cache()

        # Model parameters with default settings
        if "app_feats" in trainer_config["dataloader_name"]:
            self.use_app_feats = True
        else:
            self.use_app_feats = False
        self.app_feat_length = 0  # Default setting

        # Defining other known parameters
        self.training_log_filepath = "./checkpoints/training_log.txt"
        # self.verbose_log_filepath = "./checkpoints/verbose_log.txt"
        self.testing_log_filepath = "./checkpoints/testing_log.txt"
        self.model_name = trainer_config["model_name"]
        self.curr_label = trainer_config["curr_label"]
        self.max_length = trainer_config["max_length"]
        self.batch_size = trainer_config["batch_size"]
        self.dataloader_ready = trainer_config["dataloader_ready"]
        self.max_training_steps = trainer_config["max_training_steps"]
        # self.label_filters = trainer_config["label_filters"]
        self.num_feat = [] #trainer_config["num_feat"] #### EDIT JOEY ####
        self.cat_feat = [] #trainer_config["cat_feat"] #### EDIT JOEY ####
        self.lr = trainer_config["lr"]
        # self.hinge_idx = trainer_config["hinge_idx"]
        self.use_hinge_loss = trainer_config["use_hinge_loss"]
        try:
            self.use_epsilon = trainer_config["use_epsilon"]
            self.hinge_epsilon = trainer_config["hinge_epsilon"]
        except:
            self.use_epsilon = False
            self.hinge_epsilon = None
        try:
            self.hinge_loss_fn_name = trainer_config['hinge_loss_fn_name']
        except:
            self.hinge_loss_fn_name = 'relu'
        #### EDIT JOEY ####
        # try:
        #     self.text_source = trainer_config["text_source"]
        # except:
        #     self.text_source = "claim"

        if "total_epochs" not in trainer_config.keys():
            self.total_epochs = 5
            trainer_config["total_epochs"] = 5
        else:
            self.total_epochs = trainer_config["total_epochs"]

        #### EDIT JOEY ####
        # try:
        #     self.split_years = trainer_config['split_years']
        #     if self.split_years:
        #         self.selected_years = trainer_config['years']
        # except:
        #     self.split_years = False

        # Beilei: check size before moving to gpu
        # print("################ size of hinge_lambda: ", sys.getsizeof(torch.tensor(trainer_config["hinge_lambda"])))

        # Defining eval_loss
        self.hinge_lambda = torch.tensor(trainer_config["hinge_lambda"]).to(self.device)


        self.regularization = HingeLossRegularizer(self.hinge_lambda, self.hinge_loss_fn_name)
        # self.loss = nn.NLLLoss(weight = weights)

        # Beilei Question: What is abstract paramerter?
        if "abstract" in self.model_name:
            self.abstract_model = True
        else:
            self.abstract_model = False
        self.abstract_needed_columns = [
            "applicationNumber",
            "groupArtUnitNumber",
            "applicationTypeCategory",
            "relatedDocumentData",
            "patentClassification",
            "applicantCitedExaminerReferenceIndicatorCount",
            "filingDate",
            "publicationDate",
            "claimNumberArrayDocument",
            "abstract",
            "percentile",
            "relatedApplicationNumber",
            "max_score_x",
            "mean_score",
            "max_citations",
            "max_other_citations",
            "max_article_citations",
            "claim_label_101_adjusted",
        ]

    def train_model(self):
        """
        Description:
            initialize experiment settings
        Input:
            trainer_config
        Output:
            None

        """
        # Preparing data
        if self.dataloader_ready == 0:
            self.data_prep()
            self.dataloader_ready = 1
        ###### EDIT: change to a seperate flag ######

        print()
        print()
        print("================== Running Customized Model Training ==================")
        print(self.trainer_config)
        print()
        # Checking dataloader format
        app_feats_train = None  # Default setting
        app_feats_val = None  # Default setting
        app_feats_test = None  # Default setting

        if self.use_app_feats: ###### JOEY EDIT ######
            # app_feats_train = pickle.load(
            #     open(self.dataloaders_dir + "app_feats_train.pickle", "rb")
            # )
            # app_feats_val = pickle.load(
            #     open(self.dataloaders_dir + "app_feats_val.pickle", "rb")
            # )
            # app_feats_test = pickle.load(
            #     open(self.dataloaders_dir + "app_feats_test.pickle", "rb")
            # )
            app_feats_sample = pickle.load(
                open(self.dataloaders_dir + "app_feats_sample.pickle", "rb")
            )

            print(
                "==>  [Trainer.py: train_model(self)] Shape of app_feats:",
                app_feats_sample.shape,
            )
            app_feat_length = app_feats_sample[0].shape[0]
            self.use_app_feats = True
            self.app_feat_length = app_feat_length

        # Loading in the data
        # Incoporate batch_size here if the dataloader is compatible
        if "flex" in self.dataloader_name:
            # train_dataset = pickle.load(
            #     open(self.dataloaders_dir + "train_dataloader_claims.pickle", "rb")
            # )
            #
            # validation_dataset = pickle.load(
            #     open(self.dataloaders_dir + "validation_dataloader_claims.pickle", "rb")
            # )
            #
            # test_dataset = pickle.load(
            #     open(self.dataloaders_dir + "test_dataloader_claims.pickle", "rb")
            # )
            #
            # train_dataloader = prepare_dataloader(
            #     train_dataset, RandomSampler, self.batch_size
            # )
            # validation_dataloader = prepare_dataloader(
            #     validation_dataset, SequentialSampler, self.batch_size
            # )
            # test_dataloader = prepare_dataloader(
            #     test_dataset, SequentialSampler, self.batch_size
            # ) #### EDIT JOEY ####
            pass
        else:
            train_dataloader = pickle.load(
                open(self.dataloaders_dir + "train_dataloader_claims.pickle", "rb")
            )
            validation_dataloader = pickle.load(
                open(self.dataloaders_dir + "validation_dataloader_claims.pickle", "rb")
            )
            test_dataloader = pickle.load(
                open(self.dataloaders_dir + "test_dataloader_claims.pickle", "rb")
            )

        # convert to dataloader if dataset
        if isinstance(train_dataloader, TensorDataset):
            train_dataloader = prepare_dataloader(
                train_dataloader, RandomSampler, self.batch_size
            )
            validation_dataloader = prepare_dataloader(
                validation_dataloader, SequentialSampler, self.batch_size
            )
            test_dataloader = prepare_dataloader(
                test_dataloader, SequentialSampler, self.batch_size
            )

        y_train = pickle.load(open(self.dataloaders_dir + "y_train.pickle", "rb"))

        # Training params settings
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )  # compute the class weights
        # print("Class Weights:", class_weights)
        # class_weights = np.array([7.63396696, 0.53504362])
        weights = torch.tensor(class_weights, dtype=torch.float)
        print(
            "==> [Trainer.py: train_model(self)] Current class Weights:", class_weights
        )
        
        weights = weights.to(self.device)  # push to GPU
        cross_entropy = nn.NLLLoss(weight=weights)

        # setting random seed
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # Defining model
        print(self.model_name)
        model_str = ""
        if "xlnet" in self.model_name:
            self.trainer_config["app_feat_length"] = self.app_feat_length
            model_str = "xlnet-base-cased"
            if self.from_checkpoint != 0:
                model_str = self.checkpoint_dir + str(self.from_checkpoint)
            self.model = XLNetForPatentPrediction.from_pretrained(
                model_str, trainer_config=self.trainer_config
            )
        elif "bert" in self.model_name:
            print("loading bert")
            self.trainer_config["app_feat_length"] = self.app_feat_length
            model_str = "bert-base-uncased"
            if self.from_checkpoint != 0:
                model_str = self.checkpoint_dir + str(self.from_checkpoint)
            self.model = BertForPatentPrediction.from_pretrained(
                model_str, trainer_config=self.trainer_config
            )
        #### EDIT JOEY ####
        #new parts for mlp
        # elif "_bert2Layer_" in self.model_name:
        #     print("loading bert2Layer")
        #     self.trainer_config["app_feat_length"] = self.app_feat_length
        #     model_str = "bert-base-uncased"
        #     if self.from_checkpoint != 0:
        #         model_str = self.checkpoint_dir + str(self.from_checkpoint)
        #     self.model = Bert2LayerForPatentPrediction.from_pretrained(
        #         model_str, trainer_config=self.trainer_config
        #     )
        # elif "_bert3Layer_" in self.model_name:
        #     print("loading bert3Layer")
        #     self.trainer_config["app_feat_length"] = self.app_feat_length
        #     model_str = "bert-base-uncased"
        #     if self.from_checkpoint != 0:
        #         model_str = self.checkpoint_dir + str(self.from_checkpoint)
        #     self.model = Bert3LayerForPatentPrediction.from_pretrained(
        #         model_str, trainer_config=self.trainer_config
        #     )
        # elif "_bert4Layer_" in self.model_name:
        #     print("loading bert4Layer")
        #     self.trainer_config["app_feat_length"] = self.app_feat_length
        #     model_str = "bert-base-uncased"
        #     if self.from_checkpoint != 0:
        #         model_str = self.checkpoint_dir + str(self.from_checkpoint)
        #     self.model = Bert4LayerForPatentPrediction.from_pretrained(
        #         model_str, trainer_config=self.trainer_config
        #     )
        # # New part for BigBird
        # elif "bigbird" in self.model_name:
        #     self.trainer_config["app_feat_length"] = self.app_feat_length
        #     model_str = "google/bigbird-roberta-base"
        #     if self.from_checkpoint:
        #         model_str = self.checkpoint_dir + str(self.from_checkpoint)
        #     self.model = BigBirdForPatentPrediction.from_pretrained(
        #         model_str, trainer_config=self.trainer_config
        #     )
        
        self.model.cuda(self.device_num)

        # Store the average loss after each epoch so we can plot them.
        loss_values = []
        # param_optimizer = list(self.model.named_parameters())
        # no_decay = ['bias', 'gamma', 'beta']
        total_steps = len(train_dataloader) * self.total_epochs
        print('-----------------')
        print('loaded dataloader!')
        print('total steps:', total_steps)
        print('-----------------')
        optimizer = AdamW(self.model.parameters(), self.lr, eps=1e-8)

        # Training settings
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # Default value in run_glue.py
            num_training_steps=total_steps,
        )

        epoch_start = 0
        all_auc = []
        # if not training from scratch, resume from the previous checkpoint
        if self.from_checkpoint != 0:
            checkpoint_path = (
                self.checkpoint_dir + str(self.from_checkpoint) + "/training.pt"
            )
            checkpoint = torch.load(checkpoint_path)
            optimizer.load_state_dict(checkpoint["optim_state_dict"])
            scheduler.load_state_dict(checkpoint["sched_state_dict"])
            epoch_start = checkpoint["epoch"]
            all_auc = checkpoint["all_auc"]

        max_training_steps = total_steps
        if self.max_training_steps != 0:
            max_training_steps = self.max_training_steps

        ################ Deploy Training ################
        loss_values = []
        best_acc = 0.0
        counter = 0

        model = self.model
        print()
        print(" => Training... ")
        for epoch_i in range(epoch_start, self.total_epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            print()
            print("  => Epoch {:} / {:}".format(epoch_i + 1, self.total_epochs))

            t0 = time.time()
            total_loss = 0
            step = 0
            # log_file = open(self.verbose_log_filepath, "a")
            for batch in tqdm(train_dataloader):
                first_batch = True
                if self.use_app_feats:
                    (
                        b_input_ids,
                        b_input_mask,
                        b_labels,
                        b_app_feats,
                        b_app_num,
                        b_claim_idx,
                    ) = batch
                else:
                    b_input_ids, b_input_mask, b_labels, b_app_num, b_claim_idx = batch
                    b_app_feats = torch.tensor([])
                # print(b_app_feats.shape)

                model.zero_grad()
                model.train()

                outputs = model(
                    input_ids=b_input_ids.to(self.device),
                    attention_mask=b_input_mask.to(self.device),
                    app_feats=b_app_feats.to(self.device),
                )
                # b_input_ids.cpu()

                # logits = outputs[0]
                if self.use_hinge_loss:
                    # print(self.regularization(outputs[0], outputs[2]))
                    # print(self.regularization(outputs[0], outputs[2]).shape)
                    # print(cross_entropy(outputs[0], b_labels.cuda(self.device_num)))
                    if self.use_epsilon:
                        loss = cross_entropy(
                            outputs[0], b_labels.cuda(self.device_num)
                        ) + self.regularization(outputs[0], outputs[2]) / self.hinge_epsilon
                    else:
                        loss = cross_entropy(
                            outputs[0], b_labels.cuda(self.device_num)
                        ) + self.regularization(outputs[0], outputs[2])
                    # if first_batch:
                    #     log_file.write(str(self.regularization(outputs[0], outputs[2]).item()) + ' ')
                    #     first_batch = False
                else:
                    loss = cross_entropy(outputs[0], b_labels.cuda(self.device_num))

                total_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                step += 1

                # early stopping
                if (step > max_training_steps):  # stop if there is a specified max_training_steps
                    break
                ###### TMP EDIT ######
                # break
            # log_file.close()
            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)
            # avg_train_loss = total_loss / (counter)

            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)
            print("   => Average training loss: {0:.2f}".format(avg_train_loss))
            print(
                "   => Training epcoh took: {:}".format(format_time(time.time() - t0))
            )

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.
            print("  => Running Validation...")
            t0 = time.time()
            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()
            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            all_label = []
            all_pred = []
            all_score = []

            # Evaluate data for one epoch
            for batch in tqdm(validation_dataloader):
                if self.use_app_feats:
                    (
                        b_input_ids,
                        b_input_mask,
                        b_labels,
                        b_app_feats,
                        b_app_num,
                        b_claim_idx,
                    ) = batch
                else:
                    b_input_ids, b_input_mask, b_labels, b_app_num, b_claim_idx = batch
                    b_app_feats = torch.tensor([])

                with torch.no_grad():
                    outputs = model(
                        input_ids=b_input_ids.cuda(self.device_num),
                        attention_mask=b_input_mask.cuda(self.device_num),
                        app_feats=b_app_feats.cuda(self.device_num),
                    )

                # logits = outputs
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to("cpu").numpy()

                # Calculate the accuracy for this batch of test sentences.
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                for i in np.argmax(logits, axis=1).flatten():
                    all_pred.append(i)
                for i in label_ids.flatten():
                    all_label.append(i)
                for i in logits:
                    all_score.append(math.e ** (i[1]))

                # Accumulate the total accuracy.
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

                ###### TMP EDIT ######
                # break
            # Report the final accuracy for this validation run.
            print("   => Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            print("   => Validation took: {:}".format(format_time(time.time() - t0)))
            print("   => Output analysis:", output_analysis(all_pred, all_label))

            # Save checkpoints
            output_dir = self.checkpoint_dir + str(epoch_i + 1)
            model.save_pretrained(output_dir)

            optim_state_dict = optimizer.state_dict()
            sched_state_dict = scheduler.state_dict()

            # Save state dict
            torch.save(
                {
                    "epoch": epoch_i + 1,
                    "optim_state_dict": optim_state_dict,
                    "sched_state_dict": sched_state_dict,
                    "all_auc": all_auc
                },
                output_dir + "/training.pt",
            )

            # Logging
            now = datetime.datetime.now()
            train_logging_str = str(now) + "  Output analysis: "
            output_dict = output_analysis(all_pred, all_label)
            for key in output_dict.keys():
                train_logging_str += key + ": " + str(output_dict[key]) + ", "

            # AUC Score!
            auc_score = round(metrics.roc_auc_score(all_label, all_score), 4)
            print("   => Current validation AUC:", auc_score)
            print()
            all_auc.append(auc_score)

            # Logging!
            log_file = open(self.training_log_filepath, "a")
            log_file.write("\n")
            log_file.write(
                "Model: " + self.model_name + " hinge_fn: " + self.hinge_loss_fn_name + ", epoch: " + str(epoch_i + 1) + "\n"
            )
            log_file.write(train_logging_str + "\n")
            log_file.write(
                str(now) + "  Current validation AUC: " + str(auc_score) + "\n"
            )
            log_file.write("\n")
            log_file.close()
            # break

        print()
        print(" => [trainer.py: train_model()] Training complete!")

        self.best_auc_idx = np.argmax(all_auc) + 1
        checkpoint_path = self.checkpoint_dir + str(self.best_auc_idx)
        print(
            " => [trainer.py: train_model()] Final AUC Score (ROC) on testset:",
            self.run_single_test(test_dataloader, checkpoint_path),
        )
        return 0

    def run_single_test(
        self, test_dataloader, checkpoint_path, return_prediction_dict=False
    ):
        if "bert" in self.model_name:
            model = BertForPatentPrediction.from_pretrained(
                checkpoint_path, trainer_config=self.trainer_config
            )
        #### EDIT JOEY ####
        # elif "_xlnet_" in self.model_name:
        #     model = XLNetForPatentPrediction.from_pretrained(
        #         checkpoint_path, trainer_config=self.trainer_config
        #     )
        # # New part for bigbird
        # elif "_bigbird_" in self.model_name:
        #     model = BigBirdForPatentPrediction.from_pretrained(
        #         checkpoint_path, trainer_config=self.trainer_config
        #     )
        # elif "_bert2Layer_" in self.model_name:
        #     model = Bert2LayerForPatentPrediction.from_pretrained(
        #         checkpoint_path, trainer_config=self.trainer_config
        #     )
        # elif "_bert3Layer_" in self.model_name:
        #     model = Bert3LayerForPatentPrediction.from_pretrained(
        #         checkpoint_path, trainer_config=self.trainer_config
        #     )
        # elif "_bert4Layer_" in self.model_name:
        #     model = Bert4LayerForPatentPrediction.from_pretrained(
        #         checkpoint_path, trainer_config=self.trainer_config
        #     )
        else:
            raise Exception("model not defined")
        model.cuda(self.device_num)
        model.eval()

        print()
        print(" => Running Testing...")
        t0 = time.time()

        nb_eval_steps, nb_eval_examples = 0, 0
        all_label = []
        all_pred = []
        all_score = []
        all_prediction_dict = {}
        first_batch = True
        # Evaluate data for one epoch
        for batch in tqdm(test_dataloader):

            if self.use_app_feats:
                (
                    b_input_ids,
                    b_input_mask,
                    b_labels,
                    b_app_feats,
                    b_app_num,
                    b_claim_idx,
                ) = batch
            else:
                b_input_ids, b_input_mask, b_labels, b_app_num, b_claim_idx = batch
                b_app_feats = torch.tensor([])

            # print(b_input_ids.shape, b_app_feats.shape)

            with torch.no_grad():
                outputs = model(
                    input_ids=b_input_ids.cuda(self.device_num),
                    attention_mask=b_input_mask.cuda(self.device_num),
                    app_feats=b_app_feats.cuda(self.device_num),
                )
                # b_input_ids.detach().cpu()

            # logits = outputs
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            # print(outputs[-1])
            # print(len(outputs))
            # print(outputs)
            bert_rep = outputs[-1].cpu().numpy()

            tmp_score = []
            tmp_pred = []
            for i in np.argmax(logits, axis=1).flatten():
                all_pred.append(i)
                tmp_pred.append(i)
            for i in label_ids.flatten():
                all_label.append(i)
            for i in logits:
                tmp_score.append(math.e ** (i[1]))
                all_score.append(math.e ** (i[1]))

            claim_id = list(zip(b_app_num.numpy(), b_claim_idx.numpy()))
            for i in range(len(claim_id)):
                all_prediction_dict[claim_id[i]] = [
                    tmp_pred[i],
                    tmp_score[i],
                    bert_rep[i],
                ]
            # if first_batch:
            #     print([all_prediction_dict[key][1] for key in list(all_prediction_dict.keys())[:5]])
            #     print(all_score[:5])
            #     print(claim_id[:5])
            #     first_batch = False

        auc_score = round(metrics.roc_auc_score(all_label, all_score), 4)
        output_dict = output_analysis(all_pred, all_label)

        test_logging_str = ""
        for key in output_dict.keys():
            test_logging_str += key + ": " + str(output_dict[key]) + ", "

        print("  => Testing took: {:}".format(format_time(time.time() - t0)))
        print(
            "  => Transformer output analysis:", test_logging_str, "; AUC:", auc_score
        )

        now = datetime.datetime.now()
        with open(self.testing_log_filepath, "a") as log_file:
            to_write = (
                "Model: "
                + self.model_name
                + "\n"
                + str(now)
                + "  Output analysis: "
                + test_logging_str
                + "\n"
                + str(now)
                + "  Current testing AUC: "
                + str(auc_score)
                + "\n"
            )
            log_file.write(to_write)

        if return_prediction_dict:
            return auc_score, all_prediction_dict
        return auc_score

    def data_prep(self):
        """
        Prepare dataloaders for training specified model
            - raw data - after exploding
                - File path: “/data4/xuanyu/full_data_2021/raw/”
                    - Features:
                        "dataset": "train" or "val" or "test" or “train_expanded”
                        "application_number": application number (int)
                        "claim_input": claim texts
                        curr_label: trainer_config.curr_label (very likely is "claim_label_102")
            - Config
                - Model_name
        """
        print()
        print()
        print("================== Running Data Preparation ==================")
        print()

        if not self.exploded:
            all_data = pd.read_csv(
                self.raw_data_path + "full_data.expanded.fixed.csv",
                sep="\t",
                low_memory=False,
            )
            exploded_df = explode(all_data)
            # print(exploded_df.columns)
            exploded_df.to_csv(
                self.raw_data_path + "full_data.expanded.exploded.csv",
                sep="\t",
                index=False,
            )
            return 1
        # print(self.raw_data_path + 'full_data.main.exploded.csv')

        if not self.split_years:
            # Check raw data location
            all_claims_df = pd.read_csv(
                self.raw_data_path + "full_data.expanded.final.csv",
                sep="\t",
                low_memory=False,
            )
        else:
            all_claims_df = pd.read_csv(
                self.raw_data_path + "/year_split/sub_full_df_" + self.selected_years +".csv",
                sep="\t",
                low_memory=False,
            )

        # Filtering
        # filters = self.label_filters
        # if isinstance(filters, list) and len(filters) > 0:
        #     all_claims_df = filter_by_label(all_claims_df, filters)

        # split_dataset
        cut_offs = [30, 60, 80]
        if self.testing:
            cut_offs = [30, 72, 86]

        if self.abstract_model:
            all_claims_df = (
                all_claims_df[self.abstract_needed_columns]
                .drop_duplicates()
                .reset_index(drop=True)
                .rename(columns={"abstract": "claim_input"})
            )
            all_claims_df["claim_idx"] = 0
        all_claims_df = split_dataset(all_claims_df, cut_offs)
        train_df = all_claims_df.query('dataset == "train"').reset_index(drop=True)
        val_df = all_claims_df.query('dataset == "val"').reset_index(drop=True)
        test_df = all_claims_df.query('dataset == "test"').reset_index(drop=True)

        # dataloader dir
        os.system("mkdir -p " + self.dataloaders_dir)

        # features to use
        num_feat = (
            self.num_feat
        )  # ['bp_4_skewness', 'bp_3_skewness', 'lexical_diversity'] + citation_feat
        cat_feat = self.cat_feat  # ['patent_class']

        tmp_use_app_feats = False
        if len(num_feat + cat_feat) != 0:
            tmp_use_app_feats = True

            # Prepare perplexity features as needed
            if "bp_4_skewness" in num_feat and "bp_4_skewness" not in train_df.columns:
                print(
                    "==> [Trainer.py: data_prep(self)] Preparing perplexity features.."
                )

                # Calculating bigram language model
                print(
                    "==> [Trainer.py: data_prep(self)] Preparing bigram language model.."
                )
                corpus = " ".join(train_df.claim_input.values)
                s = clean_transcript(corpus)
                words = [
                    token
                    for token in tqdm(s.split())
                    if token != ""
                    and token not in string.punctuation
                    and token not in stop_words
                ]
                word2freq = pd.Series(words).value_counts()
                unknown_words = set(
                    list(word2freq.loc[word2freq.apply(lambda s: s < 50)].index)
                )  # filter out the combination with less than 50 appearances
                cfreq_2gram = nltk.ConditionalFreqDist(
                    nltk.bigrams([w for w in words if w not in unknown_words])
                )
                vocab_size = len(set(words) - unknown_words)
                print("==> [Trainer.py: data_prep(self)] Bigram language model ready!")

                # Calculating bp features
                print(
                    "==> [Trainer.py: data_prep(self)] Calculating bp feautres for all_claims_df..."
                )
                train_df = (
                    add_boilerplate_feature(
                        train_df, cfreq_2gram, vocab_size, word2freq
                    )
                    .query("bp_3_iqr != -1")
                    .reset_index(drop=True)
                )
                val_df = (
                    add_boilerplate_feature(val_df, cfreq_2gram, vocab_size, word2freq)
                    .query("bp_3_iqr != -1")
                    .reset_index(drop=True)
                )
                test_df = (
                    add_boilerplate_feature(test_df, cfreq_2gram, vocab_size, word2freq)
                    .query("bp_3_iqr != -1")
                    .reset_index(drop=True)
                )
                print(
                    "==> [Trainer.py: data_prep(self)] Done calculating bp feautres for all_claims_df!"
                )

            # Prepare lexical diversity feature
            if (
                "lexical_diversity" in num_feat
                and "lexical_diversity" not in train_df.columns
            ):

                def append_lexical_diversity(df):
                    assert (
                        "abstract" in df.columns
                    ), '  => [data_prep()] Require "abstract" column in data!!'
                    lexical_diversity = []
                    for a in tqdm(df.abstract.values):
                        lexical_diversity.append(cal_lexical_diversity(a))
                    df["lexical_diversity"] = lexical_diversity
                    return df

                if "lexical_diversity" not in train_df.columns:
                    num_feat.remove("lexical_diversity")
                else:
                    train_df = append_lexical_diversity(train_df)
                    val_df = append_lexical_diversity(val_df)
                    test_df = append_lexical_diversity(test_df)

            # Prepare patent class feature
            if "patent_class" in cat_feat and "patent_class" not in train_df.columns:

                def append_patent_class(df):
                    assert (
                        "patentClassification" in df.columns
                    ), '  => [data_prep()] Require "abstract" column in data!!'
                    patent_class = []
                    for a in tqdm(df.patentClassification.values):
                        patent_class.append(handle_patentClassification(a))
                    df["patent_class"] = patent_class
                    return df

                if "patentClassification" not in train_df.columns:
                    cat_feat.remove("patent_class")
                else:
                    train_df = append_patent_class(train_df)
                    val_df = append_patent_class(val_df)
                    test_df = append_patent_class(test_df)

            # Check if all the required features are presented in the train_df
            for feat in num_feat + cat_feat:
                if feat not in train_df.columns:
                    print("==> [Error!!] Some reqtmyuired features are missing!")
                    print("     EXPECTED FEATURES:", (num_feat + cat_feat))
                    print("     EXISTING FEATURES:", list(train_df.columns))
                    print("     STOP!!")
                    return 0

            # Pre-processing pipeline
            if "minmax" in self.dataloader_name:
                print("using minmax scaler")
                num_trans = Pipeline(steps=[("scaler", MinMaxScaler())])
            else:
                print("using standard scaler")
                num_trans = Pipeline(steps=[("scaler", StandardScaler())])
            cat_trans = Pipeline(
                steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
            )
            col_trans = ColumnTransformer(
                transformers=[
                    ("num", num_trans, num_feat),
                    ("cat", cat_trans, cat_feat),
                ]
            )

            X_train, y_train = (
                train_df.loc[:, num_feat + cat_feat],
                train_df[self.curr_label].values,
            )

            col_trans.fit(train_df[num_feat + cat_feat])

            app_feats_train = col_trans.transform(
                train_df[num_feat + cat_feat].fillna(0)
            )
            app_feats_val = col_trans.transform(val_df[num_feat + cat_feat].fillna(0))
            app_feats_test = col_trans.transform(test_df[num_feat + cat_feat].fillna(0))
            try:
                app_feats_train = app_feats_train.todense()
                app_feats_val = app_feats_val.todense()
                app_feats_test = app_feats_test.todense()
            except:
                pass

            app_feats_train = torch.tensor(
                [np.array(i).flatten() for i in app_feats_train]
            )
            app_feats_val = torch.tensor([np.array(i).flatten() for i in app_feats_val])
            app_feats_test = torch.tensor(
                [np.array(i).flatten() for i in app_feats_test]
            )
            app_feats_sample = torch.tensor([np.array(app_feats_train[0]).flatten()])
        else:
            app_feats_train = None
            app_feats_val = None
            app_feats_test = None

        claims_train = train_df.claim_input.values
        y_train = train_df[self.curr_label].values
        claims_val = val_df.claim_input.values
        y_val = val_df[self.curr_label].values
        claims_test = test_df.claim_input.values
        y_test = test_df[self.curr_label].values

        # Tokenization
        if "bert" in self.model_name:
            TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
        elif "xlnet" in self.model_name:
            TOKENIZER = XLNetTokenizer.from_pretrained(
                "xlnet-base-cased", do_lower_case=True
            )
        # New Part for BigBird
        elif "bigbird" in self.model_name:
            TOKENIZER = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

        # Construct claim_id list
        train_app_nums = [int(i) for i in train_df.applicationNumber.values]
        val_app_nums = [int(i) for i in val_df.applicationNumber.values]
        test_app_nums = [int(i) for i in test_df.applicationNumber.values]

        train_claim_idx = [int(i) for i in train_df.claim_idx.values]
        val_claim_idx = [int(i) for i in val_df.claim_idx.values]
        test_claim_idx = [int(i) for i in test_df.claim_idx.values]

        # print(y_train)
        X_train_tensor = tokenize(
            claims_train,
            y_train,
            TOKENIZER,
            self.max_length,
            train_app_nums,
            train_claim_idx,
            app_feats_train,
        )
        X_val_tensor = tokenize(
            claims_val,
            y_val,
            TOKENIZER,
            self.max_length,
            val_app_nums,
            val_claim_idx,
            app_feats_val,
        )
        X_test_tensor = tokenize(
            claims_test,
            y_test,
            TOKENIZER,
            self.max_length,
            test_app_nums,
            test_claim_idx,
            app_feats_test,
        )

        pickle.dump(
            X_train_tensor,
            open(self.dataloaders_dir + "train_dataloader_claims.pickle", "wb"),
            protocol=4,
        )
        pickle.dump(
            X_val_tensor,
            open(self.dataloaders_dir + "validation_dataloader_claims.pickle", "wb"),
            protocol=4,
        )
        pickle.dump(
            X_test_tensor,
            open(self.dataloaders_dir + "test_dataloader_claims.pickle", "wb"),
            protocol=4,
        )

        if tmp_use_app_feats:
            pickle.dump(
                app_feats_sample,
                open(self.dataloaders_dir + "app_feats_sample.pickle", "wb"),
            )

        pickle.dump(y_train, open(self.dataloaders_dir + "y_train.pickle", "wb"))
        print("=> [Trainer.py: data_prep(self)] Done data preparation!")
        return 0


################ Training Util Functions ################
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def output_analysis(pred, targets):
    pred = np.array(pred)
    targets = np.array(targets)
    acc = sum(pred == targets) / len(pred)
    tn, fp, fn, tp = confusion_matrix(targets, pred).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    tnr = tn / (tn + fp)
    npv = tn / (tn + fn)

    return {
        "acc": round(100 * acc, 2),
        "precision": round(100 * precision, 2),
        "recall": round(100 * recall, 2),
        "tnr": round(100 * tnr, 2),
        "npv": round(100 * npv, 2),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
