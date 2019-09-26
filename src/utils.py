import logging
import os
import random
from collections import Counter

import GPUtil
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Sampler

logging.basicConfig(format='%(asctime)s  -  %(name)s  -  %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, extra=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.extra = extra


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    if len(tokens_a) > max_length:
        tokens_a = tokens_a[:max_length]
        tokens_b = []
        return tokens_a, tokens_b
    total_length = len(tokens_a) + len(tokens_b)
    if total_length > max_length:
        tokens_b = tokens_b[total_length - max_length:]
    while True:
        cnt = 0
        if len(tokens_b) and tokens_b[0] not in ["?", "!", ".", ",", ";", ""]:
            tokens_b.pop(0)
            cnt += 1
        else:
            break
        if cnt > 10:
            print(tokens_b)
    return tokens_a, tokens_b


class CondProcessor:
    """Processor for the condescension dataset"""

    def __init__(self, use_quoted=True, use_context=False):
        assert use_quoted or use_context
        self.use_quoted = use_quoted
        self.use_context = use_context

    def get_train_examples(self, data_dir, filename, sampling_strategy=-1):
        """See base class."""
        logger.info("Get train data")
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, filename)), "train", sampling_strategy=sampling_strategy)

    def get_dev_examples(self, data_dir, filename):
        """See base class."""
        logger.info("Get dev data")
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, filename)), "dev")

    def get_test_examples(self, data_dir, filename):
        """See base class."""
        logger.info("Get test data")
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, filename)), "test")

    @classmethod
    def get_labels(cls):
        """See base class."""
        return ["0", "1"]

    @classmethod
    def _read_jsonl(cls, input_file):
        """Reads a tab separated value file."""
        logger.debug("trying to load pickle file %s" % input_file)
        df = pd.read_json(input_file, orient='records', lines=True)
        return df

    def _create_examples(self, df, set_type, sampling_strategy=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        if set_type == "train":
            cnt = Counter(df.label)
            if cnt[True] != cnt[False]:
                logger.info(f'training dataset: {cnt}')
                if sampling_strategy == -1:
                    logger.info('no oversampling')
                    pass
                else:
                    logger.info(f"setting sampling strategy to {sampling_strategy}")
                    ros = RandomOverSampler(random_state=42, sampling_strategy=sampling_strategy)
                    ids, _ = ros.fit_resample(np.arange(len(df)).reshape(-1, 1), df.label)
                    df = df.iloc[ids.reshape(-1)]
                    logger.info(f'Now training dataset: {Counter(df.label)}')
        for idx, row in enumerate(df.itertuples()):
            if self.use_quoted:
                text_a = row.quotedpost
                text_b = row.post[:row.start_offset] if self.use_context else None
            else:
                text_a = row.post[:row.start_offset]
                text_b = None

            label = 1 if row.label is True else 0
            guid = "%s-%s" % (set_type, idx)

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode='classification',
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            tokens_a, tokens_b = _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        #         print(len(input_ids), max_seq_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[str(example.label)]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def set_seed(args):
    logger.info(f'setting seed to {args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class SortedBatchSampler(Sampler):

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l, np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)


def gpu_util():
    """
    As of Aug. 24th, 2019, the official GPUtil package (if installed using `pip install gputil`) does not come with
    the correct functionality to show GPU util with customized `attrList`. You may want to download from github and
    install from the source code.
    """

    GPUtil.showUtilization(attrList=[[{'attr': 'id', 'name': 'ID'},
                                      {'attr': 'name', 'name': 'Name', 'transform': lambda x: x.replace("GeForce", "")},
                                      {'attr': 'load', 'name': 'GPU util.', 'suffix': '%',
                                       'transform': lambda x: x * 100, 'precision': 0},
                                      {'attr': 'memoryUtil', 'name': 'Mem. util.', 'suffix': '%',
                                       'transform': lambda x: x * 100, 'precision': 0}],
                                     [{'attr': 'memoryTotal', 'name': 'Mem. total', 'suffix': 'MB', 'precision': 0},
                                      {'attr': 'memoryUsed', 'name': 'Mem. used', 'suffix': 'MB', 'precision': 0},
                                      {'attr': 'memoryFree', 'name': 'Mem. free', 'suffix': 'MB', 'precision': 0}]]
                           )
