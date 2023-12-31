# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from datetime import datetime as dt
import logging
from pathlib import Path
import sys
import json

import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler

from transformers import T5Tokenizer

from generators.fusion_in_decoder.fid.options import Options
from generators.fusion_in_decoder.fid.data import set_data, Collator
from generators.fusion_in_decoder.fid.model import FiDT5
from generators.fusion_in_decoder.fid.evaluation import calc_em
import generators.fusion_in_decoder.fid.slurm
from generators.fusion_in_decoder.fid import util


DATETIME = dt.now().strftime("%Y%m%d-%H%M")
FILENAME = __file__.split("/")[-1].rsplit(".", 1)[0]

logging.basicConfig(
    format="%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(filename=f".log_{FILENAME}_{DATETIME}")
    ],
)
logger = logging.getLogger(__name__)


def evaluate(args, dataset, collator, tokenizer, model):
    sampler = SequentialSampler(dataset) 
    dataloader = DataLoader(
        dataset, 
        sampler = sampler, 
        batch_size = args.per_gpu_batch_size,
        num_workers = args.num_workers,
        collate_fn = collator
    )

    model.eval()
    total = 0
    eval_em = []
    model = model.module if hasattr(model, "module") else model

    if args.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage() 
    if args.write_results:
        write_path = Path(args.checkpoint_dir) / args.name / "test_results"
        os.makedirs(write_path, exist_ok=True)
        fw = open(write_path / ("%d.txt" % args.global_rank), "w")

    with torch.no_grad():
        for idx, batch in enumerate(dataloader, start=1):
            if args.write_crossattention_scores:
                model.reset_score_storage()
            qids, target_ids, target_masks, passage_ids, passage_masks = batch
            outputs = model.generate(
                input_ids=passage_ids.cuda(),
                attention_mask=passage_masks.cuda(),
                max_length=50,
                return_dict_in_generate=True,
                output_scores=True,
            )

            if args.write_crossattention_scores:
                crossattention_scores = model.get_crossattention_scores(passage_masks.cuda())

            for bix, output in enumerate(outputs.sequences):
                logger.info(
                    f"qid: {dataset.data[qids[bix]]['id']}, "
                    f"position: {dataset.data[qids[bix]]['position']}, "
                    f"pred_candidate: {tokenizer.decode(output, skip_special_tokens=True)}, "
                    f"sequences_prob: {torch.exp(outputs.sequences_scores[bix]) * 100}"
                )

                # 文生成スコアから算出される生成確率が事前に設定した閾値以下である、または空文字が解答候補の場合は"None"にする
                if (torch.exp(outputs.sequences_scores[bix])*100 < args.threshold_probability) or (not tokenizer.decode(output, skip_special_tokens=True)):
                    pred = None
                else:
                    pred = tokenizer.decode(output, skip_special_tokens=True)

                example = dataset.data[qids[bix]]
                if "answers" in example:
                    eval_em.append(calc_em(pred, example["answers"]))
                total += 1

                reader_prediction = {
                    "qid": str(example["id"]),
                    "position": example["position"],
                    "prediction": pred,
                    "generated": tokenizer.decode(output, skip_special_tokens=True),
                    "score": torch.exp(outputs.sequences_scores[bix]).item() * 100
                }

                if args.write_results:
                    print(json.dumps(reader_prediction, ensure_ascii=False), file=fw)
                    # fw.write(f'{{"qid": "{str(example["id"])}", "position": {example["position"]}, "prediction": {pred}}}\n')
                else:
                    print(json.dumps(reader_prediction, ensure_ascii=False))
                    # print(f'{{"qid": "{str(example["id"])}", "position": {example["position"]}, "prediction": {pred}}}')
                if args.write_crossattention_scores:
                    for j in range(passage_ids.size(1)):
                        example["ctxs"][j]["score"] = crossattention_scores[bix, j].item()

            if idx % args.eval_print_freq == 0:
                logger.info(f"Process rank: {args.global_rank}, {idx}/{len(dataloader)}")
                if len(eval_em) > 0:
                    logger.info(f"  - Ave.EM = {np.mean(eval_em):.6f}")

    if args.is_distributed:
        torch.distributed.barrier()

    logger.info(f"Process rank: {args.global_rank}, total: {total}")
    if len(eval_em) > 0:
        ave_em = np.mean(eval_em)
        logger.info(f"average = {ave_em:.6f}")
        score, total = util.weighted_average(ave_em, total, args)
        logger.info(f"EM {score:.6f}, Total number of example {total}")
        return score, total, reader_prediction
    else:
        return 0.0, total, reader_prediction


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    args = options.parse()
    
    if args.is_distributed:
        torch.distributed.barrier()
    generators.fusion_in_decoder.fid.slurm.init_distributed_mode(args)
    generators.fusion_in_decoder.fid.slurm.init_signal_handler()

    # Tokenizer & Model
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = FiDT5.from_pretrained(args.model_path)
    model = model.to(args.device)

    args.train_batch_size = args.per_gpu_batch_size * max(1, args.world_size)

    # Dataset
    collator = Collator(tokenizer, args.text_maxlength)
    set_data_fn = set_data(global_rank=args.global_rank, world_size=args.world_size)
    eval_dataset = set_data_fn(args.eval_data, args.n_context)

    em, total, reader_prediction = evaluate(
        args,
        eval_dataset, collator,
        tokenizer, model
    )

    if args.write_results and args.is_main:
        dir_path = Path(args.checkpoint_dir) / args.name
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "test_results").mkdir(parents=True, exist_ok=True)
        glob_path = Path(args.checkpoint_dir) / args.name / "test_results"
        write_path = Path(args.checkpoint_dir) / args.name / "final_output.jsonl"
        util.write_output(glob_path, write_path) 
    if args.write_crossattention_scores:
        util.save_distributed_dataset(eval_dataset.data, args)

