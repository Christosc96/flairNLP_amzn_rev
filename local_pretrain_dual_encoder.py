import argparse
import copy
import itertools
import json
from pathlib import Path

import flair
from flair.embeddings import (
    TransformerDocumentEmbeddings,
    TransformerWordEmbeddings,
    WordEmbeddings,
)
from flair.models import DualEncoder
from flair.trainers import ModelTrainer
from local_corpora import get_masked_fewnerd_corpus


def main(args):
    if args.cuda:
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = Path(
        f"{args.cache_path}/pretrained-dual-encoder/"
        f"{args.transformer if args.label_encoder == 'transformer' else 'glove'}"
        f"_{args.corpus}{f'-{args.fewnerd_granularity}'}-masked"
        f"_{args.lr}-{args.seed}"
    )

    # corpus = get_corpus(args.corpus, args.fewnerd_granularity, mask_ratio=args.mask_ratio)
    assert args.corpus == "fewnerd"
    corpus, kept_labels = get_masked_fewnerd_corpus(args.seed, args.fewnerd_granularity, inverse_mask=False)

    tag_type = "ner"
    label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)

    token_encoder = TransformerWordEmbeddings(args.transformer)
    if args.label_encoder == "transformer":
        label_encoder = TransformerDocumentEmbeddings(args.transformer)
    elif args.label_encoder == "glove":
        label_encoder = WordEmbeddings(args.cache_path + "/glove/glove.6B.300d.txt")
    else:
        raise Exception("Unknown label encoder.")

    model = DualEncoder(
        token_encoder=token_encoder, label_encoder=label_encoder, tag_dictionary=label_dictionary, tag_type=tag_type
    )

    trainer = ModelTrainer(model, corpus)

    trainer.fine_tune(
        save_base_path,
        learning_rate=args.lr,
        mini_batch_size=args.bs,
        mini_batch_chunk_size=args.mbs,
        max_epochs=args.epochs,
    )

    with open(save_base_path / "kept_labels.json", "w") as f:
        json.dump(kept_labels, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123, nargs="+")
    parser.add_argument("--cache_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon/flair-models")
    parser.add_argument("--corpus", type=str, default="fewnerd")
    parser.add_argument("--fewnerd_granularity", "--list", nargs="+", default="coarse")
    parser.add_argument("--label_encoder", type=str, default="transformer")
    parser.add_argument("--transformer", type=str, default="bert-base-uncased")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bs", type=int, default=10)
    parser.add_argument("--mbs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if all(isinstance(var, list) for var in [args.seed, args.fewnerd_granularity]):
        for seed, fewnerd_granularity in itertools.product(args.seed, args.fewnerd_granularity):
            config = copy.copy(args)
            config.seed = seed
            config.fewnerd_granularity = fewnerd_granularity
            main(config)
    else:
        main(args)
