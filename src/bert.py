# author: Zijian Wang
# many pieces of code were adapted from `pytorch_transformer` repo

import argparse
import glob
import json
import sys

from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import trange, tqdm

from .utils import *

import warnings

# skip numpy future warning
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(format='%(asctime)s  -  %(name)s  -  %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# List all model types here from `pytorch_transformer`. Only bert was tested.
ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)),
    ())
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}



def get_dataloader(args, processor, label_list, tokenizer, is_train=False):
    if is_train:
        logger.info('getting train dataloader')
        examples = processor.get_train_examples(args.data_dir, args.train_file,
                                                      sampling_strategy=args.sampling_strategy)
    else:
        logger.info(f'getting {"test" if args.eval_on_test else "dev"} dataloader')
        if args.eval_on_test:
            examples = processor.get_test_examples(args.data_dir, args.test_file)
        else:
            examples = processor.get_dev_examples(args.data_dir, args.dev_file)

    features = convert_examples_to_features(
        examples, label_list, args.max_seq_length, tokenizer, output_mode='classification',
        cls_token_at_end=bool(args.model_type in ['xlnet']),  # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(args.model_type in ['roberta']),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(args.model_type in ['xlnet']),  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    if is_train:
        sampler = SortedBatchSampler([sum(i) for i in all_input_mask.numpy()], args.train_batch_size)
    else:
        sampler = SequentialSampler(data)

    dataloader = DataLoader(data, sampler=sampler,
                            batch_size=max(args.n_gpu, 1) * args.per_gpu_eval_batch_size if is_train
                            else max(args.n_gpu, 1) * args.per_gpu_train_batch_size,
                            num_workers=args.n_gpu * 2, pin_memory=True, drop_last=True if is_train else False)
    return dataloader


def evaluate(eval_dataloader, model, args):
    labels = []
    preds = []

    for batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                  # XLM and RoBERTa don't use segment_ids
                  'labels': batch[3]}

        with torch.no_grad():
            outputs = model(**inputs)
            _, logits = outputs[:2]

        logits = logits.detach().cpu().numpy()
        label_ids = batch[3].to('cpu').numpy()
        pred = np.argmax(logits, axis=1)
        labels.append(label_ids)
        preds.append(pred)

    f1 = f1_score(np.concatenate(labels), np.concatenate(preds), average="macro")
    return f1

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='./data/', type=str, required=False,
                        help="The input data dir. Should contain the jsonl files for the task.")
    parser.add_argument("--model_type", default='bert', type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='bert-base-cased', type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default='test', type=str, required=False)
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=400, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--eval_on_test", action='store_true', help="Whether to evaluate on test (final) or dev set")

    parser.add_argument("--use_quoted",
                        action='store_true',
                        help="Whether to use quoted part as dataset")

    parser.add_argument("--use_context",
                        action='store_true',
                        help="Whether to use context part as dataset")

    parser.add_argument("--tag",
                        default='exp_0',
                        type=str,
                        help="The name of this experiment")

    parser.add_argument("--sampling_strategy", type=float, default=-1,
                        help='The oversampling ratio for training dataset. 1 means oversampling to balance and -1 means '
                             'no oversampling')

    parser.add_argument("--train_file", type=str, default="imbalanced_train.jsonl")
    parser.add_argument("--dev_file", type=str, default="imbalanced_dev.jsonl")
    parser.add_argument("--test_file", type=str, default="imbalanced_test.jsonl")

    args = parser.parse_args()

    if args.do_train and args.output_dir is None:
        setattr(args, 'output_dir', os.path.join('models', args.task_name, args.tag))

    if args.do_train:
        logger.info(f"Model will be saved to {args.output_dir}")
    else:
        logger.info(f"Model will be loaded to {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    logger.warning(f"Device: {device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    if not os.path.exists(args.output_dir) or os.listdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    processor = CondProcessor(use_quoted=args.use_quoted, use_context=args.use_context)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()

    tb_writer = SummaryWriter(logdir=f'runs/{args.task_name}_{args.tag}')

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    args.output_mode = 'classification'

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    model.to(device)

    eval_f1s = []

    if args.do_train:

        # get data
        train_dataloader = get_dataloader(args, processor, label_list, tokenizer, is_train=True)

        # get training setting

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader.dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size = %d",
                    args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        model.zero_grad()

        # reset seed before training
        set_seed(args)

        # train
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()

            for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), mininterval=5):

                min_len = torch.min(torch.tensor(args.max_seq_length), torch.max(torch.sum(batch[1], dim=1)) + 2)
                batch = [
                    t.to(device, non_blocking=True) if len(t.shape) == 1 else t[:, :min_len].contiguous().to(device,
                                                                                                             non_blocking=True)
                    for t in batch]

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}
                outputs = model(**inputs)

                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule

                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

            tb_writer.close()

        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

    if args.do_eval:
        eval_dataloader = get_dataloader(args, processor, label_list, tokenizer, is_train=False)

        if args.eval_all_checkpoints:
            best_f1 = -1
            checkpoints = list(os.path.dirname(c) for c in
                               sorted(glob.glob(args.output_dir + '/checkpoint*/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)

            for checkpoint in checkpoints:
                model = model_class.from_pretrained(checkpoint)
                model.to(device)
                model.eval()
                if args.n_gpu > 1:
                    model = torch.nn.DataParallel(model)

                f1 = evaluate(eval_dataloader, model, args)

                logger.info(f"{checkpoint}'s F1 is {f1}")
                eval_f1s.append(f1)
                if f1 > best_f1:

                    best_f1 = f1
                    logger.info("Best F1 %s" % best_f1)
                    result = {'eval_f1': f1,
                              'ckpt': checkpoint}

                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Eval results *****")
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))

                    model_to_save = model.module if hasattr(model, 'module') else model  #
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_param_file = os.path.join(args.output_dir, "param")
                    with open(output_param_file, 'w') as f:
                        json.dump(args.__dict__, f, indent=2, sort_keys=True)

        else:
            model = model_class.from_pretrained(args.output_dir)
            model.to(device)
            model.eval()
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            f1 = evaluate(eval_dataloader, model, args)
            logger.info(f"Model's F1 is {f1}")

if __name__ == "__main__":
    main()
