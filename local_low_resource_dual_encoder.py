import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataset import Subset

import flair
from flair.embeddings import TransformerDocumentEmbeddings, TransformerWordEmbeddings
from flair.models import DualEncoder
from flair.trainers import ModelTrainer
from local_corpora import get_masked_fewnerd_corpus


def main(args):
    if args.cuda:
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = Path(
        f"{args.cache_path}/lowresource-dual-encoder/masked-models/"
        f"{args.transformer}"
        f"_{args.corpus}{f'-{args.fewnerd_granularity}'}-masked"
        f"_{args.lr}"
        f"{'_early-stopping' if args.early_stopping else ''}"
    )

    with open(f"data/fewshot/fewshot_masked-{args.corpus}-{args.fewnerd_granularity}.json", "r") as f:
        fewshot_indices = json.load(f)

    assert args.corpus == "fewnerd"
    base_corpus, kept_labels = get_masked_fewnerd_corpus(
        int(args.fewnerd_sampling_seed), args.fewnerd_granularity, inverse_mask=True
    )

    results = {}
    for k in args.k:
        results[f"{k}"] = {"results": []}
        for seed in range(5):
            flair.set_seed(seed)
            corpus = copy.copy(base_corpus)
            corpus._train = Subset(base_corpus._train, fewshot_indices[f"{k}-{args.fewnerd_sampling_seed}-{seed}"])
            corpus._dev = Subset(base_corpus._train, [])

            tag_type = "ner"
            label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)
            label_dictionary.span_labels = True

            token_encoder = TransformerWordEmbeddings(args.transformer)
            label_encoder = TransformerDocumentEmbeddings(args.transformer)

            model = DualEncoder(
                token_encoder=token_encoder,
                label_encoder=label_encoder,
                tag_dictionary=label_dictionary,
                tag_type=tag_type,
            )

            trainer = ModelTrainer(model, corpus)

            save_path = save_base_path / f"{k}shot_{args.fewnerd_sampling_seed}-{seed}"

            result = trainer.train(
                save_path,
                learning_rate=args.lr,
                mini_batch_size=args.bs,
                mini_batch_chunk_size=args.mbs,
                max_epochs=args.epochs,
                optimizer=torch.optim.AdamW,
                train_with_dev=args.early_stopping,
                min_learning_rate=args.min_lr if args.early_stopping else 0.001,
                save_final_model=False,
            )

            results[f"{k}"]["results"].append(result["test_score"])

        def postprocess_scores(scores: dict):
            rounded_scores = [round(float(score) * 100, 2) for score in scores["results"]]
            return {"results": rounded_scores, "average": np.mean(rounded_scores), "std": np.std(rounded_scores)}

        results = {setting: postprocess_scores(result) for setting, result in results.items()}

        with open(save_base_path / "results.json", "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--cuda_device", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cache_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon/flair-models")
    parser.add_argument("--corpus", type=str, default="fewnerd")
    parser.add_argument("--fewnerd_granularity", type=str, default="coarse", nargs="+")
    parser.add_argument("--fewnerd_sampling_seed", type=int, default=10, nargs="+")
    parser.add_argument("--k", type=int, default=1, nargs="+")
    parser.add_argument("--transformer", type=str, default="bert-base-uncased")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bs", type=int, default=10)
    parser.add_argument("--mbs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stopping", type=bool, default=True)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    args = parser.parse_args()
    if all(isinstance(var, list) for var in [args.fewnerd_sampling_seed, args.fewnerd_granularity]):
        for fewnerd_sampling_seed, fewnerd_granularity in itertools.product(
            args.fewnerd_sampling_seed, args.fewnerd_granularity
        ):
            config = copy.copy(args)
            config.fewnerd_sampling_seed = fewnerd_sampling_seed
            config.fewnerd_granularity = fewnerd_granularity
            main(config)
    main(args)
