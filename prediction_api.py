import logging

import torch
from fastapi import FastAPI
import glob
import os

from retrievers.DPR.dpr.models import init_biencoder_components
from retrievers.DPR.dpr.options import (
    setup_args_gpu,
    set_encoder_params_from_state,
)
from retrievers.DPR.dpr.utils.data_utils import read_ctxs
from retrievers.DPR.dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint
)
from retrievers.DPR.dpr.indexer.faiss_indexers import (
    DenseHNSWFlatIndexer,
    DenseFlatIndexer
)
from dense_retriever import DenseRetriever, validate, save_results

from generators.fusion_in_decoder.fid.data import set_data, Collator
from generators.fusion_in_decoder.fid.slurm import init_distributed_mode, init_signal_handler
from transformers import T5Tokenizer
from generators.fusion_in_decoder.fid.model import FiDT5
from test_generator import evaluate

from argparse import Namespace


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger()


BIENCODER_CKPT_FILE = "retrievers/DPR/models/baseline/biencoder.pt"
READER_CKPT_DIR = "generators/fusion_in_decoder/models_and_results/baseline"
PASSAGE_EMBEDDINGS_FILE = "retrievers/DPR/models/baseline/embedding.pickle"
PASSAGES_FILE = "retrievers/DPR/datasets/wiki/jawiki-20220404-c400-large.tsv.gz"


def create_args():
    ## return dict instead
    args = {
        "out_file": None,
        "match": "string",
        "validation_workers": 16,
        "batch_size": 32,
        "index_buffer": 50000,
        "hnsw_index": False,
        "save_or_load_index": False,
        "pretrained_model_cfg": None,
        "encoder_model_type": None,
        "pretrained_file": None,
        "model_file": None,
        "projection_dim": 0,
        "sequence_length": 512,
        "no_cuda": False,
        "local_rank": -1,
        "fp16": False,
        "fp16_opt_level": "O1",
        "do_lower_case": False,
    }

    return args


def load_retriever(biencoder_ckpt_file: str, passage_embeddings_file: str):
    args = create_args()
    args = Namespace(**args)
    setup_args_gpu(args)

    saved_state = load_states_from_checkpoint(biencoder_ckpt_file)
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
    ctx_files_pattern = passage_embeddings_file
    input_paths = glob.glob(ctx_files_pattern)

    index_path = "_".join(input_paths[0].split("_")[:-1])
    if args.save_or_load_index and os.path.exists(index_path):
        retriever.index.deserialize(index_path)
    else:
        logger.info('Reading all passages data from files: %s', input_paths)
        retriever.index_encoded_data(input_paths, buffer_size=index_buffer_sz)
        if args.save_or_load_index:
            retriever.index.serialize(index_path)

    return retriever, tokenizer


def load_reader(reader_ckpt_dir: str, reader_base_model_name):
    args = {
        "config_file": None,
        "name": "experiment_name",
        "checkpoint_dir": "./checkpoint/",
        "model_path": None,
        "per_gpu_batch_size": 1,
        "maxload": -1,
        "num_workers": 12,
        "local_rank": -1,
        "main_port": -1,
        "seed": 0,
        "eval_step": 500,
        "save_freq": 5000,
        "eval_print_freq": 1000,
        "train_data": "none",
        "eval_data": "none",
        "model_name_or_path": "t5-base",
        "use_checkpoint": False,
        "text_maxlength": 200,
        "answer_maxlength": -1,
        "no_title": False,
        "n_context": 1,
        "write_results": False,
        "write_crossattention_scores": False,
        "threshold_probability": 85.0,
        "is_distributed": False,
    }

    args = Namespace(**args)

    if args.is_distributed:
        torch.distributed.barrier()
    init_distributed_mode(args)
    init_signal_handler()

    # Tokenizer & Model
    tokenizer = T5Tokenizer.from_pretrained(reader_base_model_name)
    model = FiDT5.from_pretrained(reader_ckpt_dir)
    model = model.to(args.device)

    args.train_batch_size = args.per_gpu_batch_size * max(1, args.world_size)

    return args, model, tokenizer


class FiDPipeline:
    def __init__(
        self,
        biencoder_ckpt_file: str,
        reader_ckpt_dir: str,
        passage_embeddings_file: str,
        passages_file: str,
        device: str = "cuda",
    ):
        # retriever
        retriever, retriever_tokenizer = load_retriever(biencoder_ckpt_file, passage_embeddings_file)
        self.retriever_module = retriever
        self.retriever_tokenizer = retriever_tokenizer

        self.validation_workers = 12
        self.match = 'string'
        self.n_docs = 100

        self.all_passages = read_ctxs(passages_file, return_dict=True)
        if len(self.all_passages) == 0:
            raise RuntimeError('No passages data found. Please specify ctx_file param properly.')

        # reader
        self.reader_base_model = "sonoisa/t5-base-japanese"
        reader_args, reader, reader_tokenizer = load_reader(reader_ckpt_dir, self.reader_base_model)
        self.reader_predict_module = reader
        self.reader_tokenizer = reader_tokenizer

        # reader args
        self.threshold_probability = 85.0
        self.text_maxlength = reader_args.text_maxlength
        self.n_context = reader_args.n_context
        self.global_rank = reader_args.global_rank
        self.world_size = reader_args.world_size
        self.per_gpu_batch_size = reader_args.per_gpu_batch_size
        self.num_workers = reader_args.num_workers
        self.write_crossattention_scores = False
        self.write_results = False
        self.eval_print_freq = 100000
        self.is_distributed = reader_args.is_distributed
        self.global_rank = reader_args.global_rank


    def predict_answer(
        self,
        qid: str,
        position: int,
        question: str,
    ) -> dict:
        # retriever
        questions_tensor = self.retriever_module.generate_question_vectors([question])
        top_ids_and_scores = self.retriever_module.get_top_docs(questions_tensor.numpy(), self.n_docs)

        questions_doc_hits = validate(self.all_passages, [[]], top_ids_and_scores, self.validation_workers,
                                      self.match, self.retriever_tokenizer, fo_acc=None)

        retrieved_data = save_results(
            self.all_passages, [qid], [str(position)], [question], [[]], top_ids_and_scores, questions_doc_hits, ""
        )

        # transform the retrieved data
        transformed_data = []
        for instance in retrieved_data:
            transformed_data.append({
                "id": instance["qid"],
                "position": instance["position"],
                "question": instance["question"],
                "target": "",
                "ctxs": instance["ctxs"]
            })

        # reader
        collator = Collator(self.reader_tokenizer, self.text_maxlength)
        eval_dataset = set_data(
            data=transformed_data,
            n_context=self.n_context,
            global_rank=self.global_rank,
            world_size=self.world_size
        )

        _, _, reader_prediction = evaluate(
            self,
            eval_dataset, collator,
            self.reader_tokenizer, self.reader_predict_module
        )

        return {"pred_answer": reader_prediction["pred_answer"], "score": reader_prediction["score"]}


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

logger.info("Loading FiDPipeline")
pipeline = FiDPipeline(
    biencoder_ckpt_file=BIENCODER_CKPT_FILE,
    reader_ckpt_dir=READER_CKPT_DIR,
    passage_embeddings_file=PASSAGE_EMBEDDINGS_FILE,
    passages_file=PASSAGES_FILE,
    device=device,
)
logger.info("Finished loading FiDPipeline")

app = FastAPI()

@app.get("/answer")
def answer(qid: str, position: int, question: str):
    prediction = pipeline.predict_answer(qid, position, question)
    return {"prediction": prediction["pred_answer"]}
