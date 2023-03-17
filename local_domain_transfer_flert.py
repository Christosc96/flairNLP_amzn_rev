import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataset import Subset

import flair
from flair.models import SequenceTagger
from flair.optim import LinearSchedulerWithWarmup
from flair.trainers import ModelTrainer
from flair.training_utils import AnnealOnPlateau
from local_corpora import get_corpus


def main(args):
    if args.cuda:
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = Path(
        f"{args.cache_path}/fewshot-flert/"
        f"{args.transformer}_{args.corpus}{args.fewnerd_granularity}_{args.lr}_{args.seed}_{args.pretrained_on}/"
    )

    with open(f"data/fewshot/fewshot_{args.corpus}{args.fewnerd_granularity}.json", "r") as f:
        fewshot_indices = json.load(f)

    base_corpus = get_corpus(args.corpus, args.fewnerd_granularity)

    results = {}
    for k in args.k:
        results[f"{k}"] = {"results": []}
        for seed in range(5):
            flair.set_seed(seed)
            corpus = copy.copy(base_corpus)
            corpus._train = Subset(base_corpus._train, fewshot_indices[f"{k}-{seed}"])
            corpus._dev = Subset(base_corpus._train, [])

            tag_type = "ner"
            label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)

            # 4. initialize fine-tuneable transformer embeddings WITH document context
            trained_model = SequenceTagger.load(args.pretrained_model_path)
            trained_model.linear = torch.nn.Linear(768, len(label_dictionary))

            # 6. initialize trainer
            trainer = ModelTrainer(trained_model, corpus)

            save_path = save_base_path / f"{k}shot_{seed}"

            # 7. run fine-tuning
            result = trainer.fine_tune(
                save_path,
                learning_rate=args.lr,
                mini_batch_size=args.bs,
                mini_batch_chunk_size=args.mbs,
                max_epochs=args.epochs,
                scheduler=AnnealOnPlateau if args.early_stopping else LinearSchedulerWithWarmup,
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
    parser.add_argument("--corpus", type=str, default="conll_03")
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--pretrained_on", type=str)
    parser.add_argument("--fewnerd_granularity", type=str, default="")
    parser.add_argument("--k", type=int, default=1, nargs="+")
    parser.add_argument("--transformer", type=str, default="bert-base-uncased")
    parser.add_argument("--use_context", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mbs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stopping", type=bool, default=True)
    parser.add_argument("--min_lr", type=float, default=5e-7)
    args = parser.parse_args()
    main(args)