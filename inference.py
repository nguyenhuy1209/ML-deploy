import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from app.SlotFilling.utils import MODEL_CLASSES, init_logger, load_tokenizer
import app.SlotFilling.config as cfg

logger = logging.getLogger(__name__)

class SlotFillingInference:
    def __init__(self, model_path, device):
        self.device = device
        self.args = self.get_args(model_path)
        self.model = self.load_model(model_path)

    
    def get_device(self, pred_config):
        return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


    def get_args(self, model_path):
        return torch.load(os.path.join(model_path, "training_args.bin"))


    def load_model(self, model_path):
        # Check whether model exists
        if not os.path.exists(model_path):
            raise Exception("Model doesn't exists! Train first!")

        try:
            intent_label_lst = cfg.INTENT_LABEL
            slot_label_lst = cfg.SLOT_LABEL
            model = MODEL_CLASSES[self.args.model_type][1].from_pretrained(
                cfg.SL_MODEL_PATH, args=self.args, intent_label_lst=intent_label_lst, slot_label_lst=slot_label_lst
            )
            model.to(self.device)
            model.eval()
            logger.info("***** Model Loaded *****")
        except Exception:
            raise Exception("Some model files might be missing...")

        return model


    def convert_input_to_tensor_dataset(
        self,
        lines,
        args,
        tokenizer,
        pad_token_label_id,
        cls_token_segment_id=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ):
        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        all_slot_label_mask = []

        for words in lines:
            tokens = []
            slot_label_mask = []
            for word in words:
                word_tokens = tokenizer.tokenize(word)
                if not word_tokens:
                    word_tokens = [unk_token]  # For handling the bad-encoded word
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

            # Account for [CLS] and [SEP]
            special_tokens_count = 2
            if len(tokens) > args.max_seq_len - special_tokens_count:
                tokens = tokens[: (args.max_seq_len - special_tokens_count)]
                slot_label_mask = slot_label_mask[: (args.max_seq_len - special_tokens_count)]

            # Add [SEP] token
            tokens += [sep_token]
            token_type_ids = [sequence_a_segment_id] * len(tokens)
            slot_label_mask += [pad_token_label_id]

            # Add [CLS] token
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids
            slot_label_mask = [pad_token_label_id] + slot_label_mask

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = args.max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)
            all_slot_label_mask.append(slot_label_mask)

        # Change to Tensor
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
        all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

        return dataset


    def predict(self, sentence):
        # load model and args
        slot_label_lst = cfg.SLOT_LABEL

        # Convert input file to TensorDataset
        pad_token_label_id = self.args.ignore_index
        tokenizer = load_tokenizer(self.args)
        dataset = self.convert_input_to_tensor_dataset(sentence, self.args, tokenizer, pad_token_label_id)

        # Predict
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=1)

        all_slot_label_mask = None
        intent_preds = None
        slot_preds = None

        for batch in tqdm(data_loader, desc="Predicting"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "intent_label_ids": None,
                    "slot_labels_ids": None,
                }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]
                outputs = self.model(**inputs)
                _, (intent_logits, slot_logits) = outputs[:2]

                # Intent Prediction
                if intent_preds is None:
                    intent_preds = intent_logits.detach().cpu().numpy()
                else:
                    intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

                # Slot prediction
                if slot_preds is None:
                    if self.args.use_crf:
                        # decode() in `torchcrf` returns list with best index directly
                        slot_preds = np.array(self.model.crf.decode(slot_logits))
                    else:
                        slot_preds = slot_logits.detach().cpu().numpy()
                    all_slot_label_mask = batch[3].detach().cpu().numpy()
                else:
                    if self.args.use_crf:
                        slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                    else:
                        slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                    all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

        intent_preds = np.argmax(intent_preds, axis=1)

        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)

        slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
        slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

        for i in range(slot_preds.shape[0]):
            for j in range(slot_preds.shape[1]):
                if all_slot_label_mask[i, j] != pad_token_label_id:
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        logger.info("Prediction Done!")
        return slot_preds_list