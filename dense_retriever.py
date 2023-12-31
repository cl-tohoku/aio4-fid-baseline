#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import argparse
import csv
import glob
import json
import logging
import os
import pickle
import sys
import time
from typing import List, Tuple, Dict, Iterator

import numpy as np
import pandas as pd

import torch
from torch import Tensor as T
from torch import nn

from retrievers.DPR.dpr.data.qa_validation import calculate_matches
from retrievers.DPR.dpr.models import init_biencoder_components
from retrievers.DPR.dpr.options import (
    add_encoder_params,
    setup_args_gpu,
    print_args,
    set_encoder_params_from_state,
    add_tokenizer_params,
    add_cuda_params
)
from retrievers.DPR.dpr.utils.data_utils import (
    Tensorizer,
    read_qas,
    read_ctxs
)
from retrievers.DPR.dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint
)
from retrievers.DPR.dpr.indexer.faiss_indexers import (
    DenseIndexer,
    DenseHNSWFlatIndexer,
    DenseFlatIndexer
)


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

Vector = np.array
DataID = object                      # data id to be registered with faiss indexer
Score = float                        # value of vector operation (inner product if IP)
DataIDs = List[DataID]               # len(DataIDs) == top-k if search top-k
Scores = List[Score]                 # len(Scores) == top-k if search top-k
Res4Query = Tuple[DataIDs, Scores]   # search results for a query


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer, index: DenseIndexer):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> T:
        """
        Convert questions (List[str]) into tensor
        :param questions: list of questions
        :return: tensor of query
        """
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                batch_token_tensors = [self.tensorizer.text_to_tensor(q) for q in
                                       questions[batch_start:batch_start + bsz]]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info('Encoded queries %d', len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info('Total encoded queries tensor %s', query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def index_encoded_data(self, vector_files: List[str], buffer_size: int = 50000):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(iterate_encoded_files(vector_files)):
            db_id, doc_vector = item
            buffer.append((db_id, doc_vector))
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info('Data indexing completed.')

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Res4Query]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors: vectors of question converted into numpy array
        :param top_docs: number of search results (top-k)
        :return: list of search indices corresponding to the query
        """
        start = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info('index search time: %f sec.', time.time() - start)
        return results


def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    """
    Get question and answers from csv file
    :param location: location of a csv file
    :return:
    """
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            yield question, answers


def validate(passages: Dict[object, Tuple[str, str]], answers: List[List[str]],
             result_ctx_ids: List[Tuple[List[object], List[float]]],
             workers_num: int, match_type: str, tokenizer, fo_acc:str=None) -> List[List[bool]]:
    """
    Evaluates answers presence and generates csv file to write validation results.
    :param passages: dict loaded from context file
    :param answers: answers imported from QA file
    :param result_ctx_ids: results of faiss search
    :param workers_num: number of parallel processes to validate results
    :param match_type: answer matching logic type
    :param fo_acc: file name to save validation results
    :return: list of results whether they have any of matched answers
    """
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type, tokenizer)
    top_k_hits = match_stats.top_k_hits
    top_k_acc = [round(v/len(result_ctx_ids), 4) for v in top_k_hits]

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    logger.info('Validation results: top k documents hits accuracy %s', top_k_acc)

    if fo_acc:
        pd.DataFrame({'top_k':range(1,len(top_k_hits)+1) ,'n_hits':top_k_hits, 'acc':top_k_acc})\
            .astype({'top_k':int, 'n_hits': int})\
            .set_index('top_k')\
            .to_csv(open(fo_acc, 'w') if fo_acc is not None else sys.stdout, sep='\t', header=True, index=True)
    else:
        logger.info('Skipped writing validation results to file. fo_acc is %s.', fo_acc)

    return match_stats.questions_doc_hits


def save_results(passages: Dict[object, Tuple[str, str]], qids: List[str], positions: List[str], questions: List[str],
                 answers: List[List[str]], top_passages_and_scores: List[Tuple[List[object], List[float]]],
                 per_question_hits: List[List[bool]], out_file: str):
    """
    Join passages text with the result ids, their questions and assigning has|no answer labels
    :param passages: dict imported from context file
    :param qids: question id loaded from QA file
    :param questions: questions loadeded from QA file
    :param answers: answers loaded from QA file
    :param top_passages_and_scores: results of faiss search
    :param per_question_hits: list of results whether they have any of matched answers
    :param out_file: file name to save results
    :return:
    """
    merged_data = []
    assert len(per_question_hits) == len(qids) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        qid = qids[i]
        position = positions[i]
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append({
            'qid': qid,
            'position': position,
            'question': q,
            'answers': q_answers,
            'ctxs': [
                {
                    'id': results_and_scores[0][c],
                    'title': docs[c]['title'],
                    'text': docs[c]['text'],
                    'score': scores[c],
                    'has_answer': hits[c],
                } for c in range(ctxs_num)
            ]
        })

    if out_file:
        with open(out_file, "w") as writer:
            writer.write(json.dumps(merged_data, indent=4, ensure_ascii=False) + "\n")
        logger.info('Saved results * scores  to %s', out_file)
    else:
        return merged_data


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    """
    Get database id and numpy array of document vectors
    :param vector_files: list of file name to get passages vectors from
    :return:
    """
    for i, file in enumerate(vector_files):
        logger.info('Reading file %s', file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Retrieve top passage for each of the datasets')
    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--qa_file', required=True, type=str, default=None,
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--ctx_file', required=True, type=str, default=None,
                        help="All passages file in the tsv format: id \\t passage_text \\t title")
    parser.add_argument('--encoded_ctx_file', type=str, default=None,
                        help='Glob path to encoded passages (from generate_dense_embeddings tool)')
    parser.add_argument('--out_file', type=str, default=None,
                        help='output .tsv file path to write results to ')
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'],
                        help="Answer matching logic type")
    parser.add_argument('--n-docs', type=int, default=200, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')

    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'
    setup_args_gpu(args)
    print_args(args)

    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)
    tokenizer = tensorizer.tokenizer

    encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16)
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')

    prefix_len = len('question_model.')
    question_encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                              key.startswith('question_model.')}
    model_to_load.load_state_dict(question_encoder_state)
    vector_size = model_to_load.get_out_size()
    logger.info('Encoder vector_size=%d', vector_size)

    index_buffer_sz = args.index_buffer
    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size)
        index_buffer_sz = -1  # encode all at once
    else:
        index = DenseFlatIndexer(vector_size)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)


    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    input_paths = glob.glob(ctx_files_pattern)

    index_path = "_".join(input_paths[0].split("_")[:-1])
    if args.save_or_load_index and os.path.exists(index_path):
        retriever.index.deserialize(index_path)
    else:
        logger.info('Reading all passages data from files: %s', input_paths)
        retriever.index_encoded_data(input_paths, buffer_size=index_buffer_sz)
        if args.save_or_load_index:
            retriever.index.serialize(index_path)

    # get questions & answers
    qids, positions, questions, question_answers = read_qas(args.qa_file)
    questions_tensor = retriever.generate_question_vectors(questions)

    # get top k results
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)

    all_passages = read_ctxs(args.ctx_file, return_dict=True)

    if len(all_passages) == 0:
        raise RuntimeError('No passages data found. Please specify ctx_file param properly.')

    fo_acc = args.out_file.replace('.json', '.tsv')
    questions_doc_hits = validate(all_passages, question_answers, top_ids_and_scores, args.validation_workers,
                                  args.match, tokenizer, fo_acc=fo_acc)

    if args.out_file:
        save_results(
            all_passages, qids, positions, questions, question_answers, top_ids_and_scores, questions_doc_hits, args.out_file
        )


if __name__ == '__main__':
    main()
