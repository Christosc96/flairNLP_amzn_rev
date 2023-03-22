import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataset import Subset

import flair
from flair.data import Dictionary
from flair.models import SequenceTagger
from flair.models.sequence_tagger_utils.crf import CRF
from flair.models.sequence_tagger_utils.viterbi import ViterbiDecoder, ViterbiLoss
from flair.optim import LinearSchedulerWithWarmup
from flair.trainers import ModelTrainer
from flair.training_utils import AnnealOnPlateau
from local_corpora import get_corpus


def bio_label_dictionary(corpus, tag_format, tag_type):
    label_dictionary = corpus.make_label_dictionary(tag_type)

    if label_dictionary.span_labels:
        # the big question is whether the label dictionary should contain an UNK or not
        # without UNK, we cannot evaluate on data that contains labels not seen in test
        # with UNK, the model learns less well if there are no UNK examples
        bio_label_dictionary = Dictionary(add_unk=False)
        for label in label_dictionary.get_items():
            if label == "<unk>":
                continue
            bio_label_dictionary.add_item("O")
            if tag_format == "BIOES":
                bio_label_dictionary.add_item("S-" + label)
                bio_label_dictionary.add_item("B-" + label)
                bio_label_dictionary.add_item("E-" + label)
                bio_label_dictionary.add_item("I-" + label)
            if tag_format == "BIO":
                bio_label_dictionary.add_item("B-" + label)
                bio_label_dictionary.add_item("I-" + label)
    else:
        bio_label_dictionary = label_dictionary
    return bio_label_dictionary


def random_initialized_classification_head(trained_model, label_dictionary):
    return torch.nn.Linear(
        trained_model.linear.in_features,
        len(label_dictionary),
        bias=True if trained_model.linear.bias is not None else False,
        device=flair.device,
    )


def match_exact(trained_model, label_dictionary):
    matching_ids = {}
    for new_idx, new_label in enumerate(label_dictionary.idx2item):
        matching_ids[new_idx] = [
            pretrained_label_id
            for pretrained_label_id, pretrained_labels in enumerate(trained_model.label_dictionary.item2idx)
            if new_label.decode("utf-8") in pretrained_labels.decode("utf-8")
        ]
    return matching_ids


def match_manual(new_label_dictionary, corpus):
    if corpus == "conll_03":
        matching = {
            b"O": [b"O"],
            b"B-person": [b"B-person"],
            b"I-person": [b"I-person"],
            b"B-location": [b"B-location", b"B-geographical social political entity"],
            b"I-location": [b"I-location", b"I-geographical social political entity"],
            b"B-miscellaneous": [b"B-nationality religion political", b"B-event"],
            b"I-miscellaneous": [b"I-nationality religion political", b"I-event"],
            b"B-organization": [b"B-organization", b"B-location"],
            b"I-organization": [b"I-organization", b"I-location"],
        }
        matching_ids = {
            new_label_dictionary.item2idx[key]: matching[key]
            for key in new_label_dictionary.item2idx.keys()
            if key in matching.keys()
        }
    elif corpus == "wnut_17":
        matching = {
            b"O": [b"O"],
            b"B-person": [b"B-person"],
            b"I-person": [b"I-person"],
            b"B-location": [b"B-location", b"B-facility", b"B-geographical social political entity"],
            b"I-location": [b"I-location", b"I-facility", b"I-geographical social political entity"],
            b"B-creative work": [b"B-work of art"],
            b"I-creative work": [b"I-work of art"],
            b"B-group": [b"B-nationality religion political"],
            b"I-group": [b"I-nationality religion political"],
            b"B-corporation": [b"B-organization"],
            b"I-corporation": [b"I-organization"],
            b"B-product": [b"B-product"],
            b"I-product": [b"I-product"],
        }
        matching_ids = {
            new_label_dictionary.item2idx[key]: matching[key]
            for key in new_label_dictionary.item2idx.keys()
            if key in matching.keys()
        }
    else:
        raise Exception()

    return matching_ids


def reuse_classification_head(trained_model, matching_mode, new_label_dictionary, corpus_name):
    new_classification_head = random_initialized_classification_head(trained_model, new_label_dictionary)
    with torch.no_grad():
        if matching_mode == "exact":
            matching_ids = match_exact(trained_model, new_label_dictionary)
        elif matching_mode == "compose":
            _matching_ids = match_manual(new_label_dictionary, corpus_name)
            matching_ids = {
                key: [trained_model.label_dictionary.item2idx[val] for val in vals]
                for key, vals in _matching_ids.items()
            }
        else:
            raise Exception()

        for new_idx, matching_idx in matching_ids.items():
            if len(matching_idx) > 1:
                weights_to_reuse = trained_model.linear.weight[matching_idx].mean(dim=0)
                if trained_model.linear.bias is not None:
                    bias_to_reuse = trained_model.linear.bias[matching_idx].mean(dim=0)
            elif len(matching_idx) == 1:
                weights_to_reuse = trained_model.linear.weight[matching_idx].squeeze()
                if trained_model.linear.bias is not None:
                    bias_to_reuse = trained_model.linear.bias[matching_idx].squeeze()
            else:
                continue

            new_classification_head.weight[new_idx] = weights_to_reuse
            if trained_model.linear.bias is not None:
                new_classification_head.bias[new_idx] = bias_to_reuse

    return new_classification_head


def main(args):
    if args.cuda:
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = Path(
        f"{args.cache_path}/reuse-weights-flert/"
        f"{args.transformer}{'-context' if args.use_context else ''}_"
        f"{args.corpus}{args.fewnerd_granularity}_"
        f"{args.lr}_{args.seed}_"
        f"{args.pretrained_on}"
        f"{'_early-stopping' if args.early_stopping else ''}"
        f"{f'_{args.matching_mode}-matching' if args.matching_mode else ''}"
        f"{f'_frozen-embeddings' if args.freeze_embeddings else ''}"
        f"{f'_decoder-lr-{args.lr * args.decoder_lr_factor}' if args.decoder_lr_factor != 1.0 else ''}"
        f"{f'_crf' if args.use_crf else ''}/"
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
            if k != 0:
                corpus._train = Subset(base_corpus._train, fewshot_indices[f"{k}-{seed}"])
            else:
                pass
            corpus._dev = Subset(base_corpus._train, [])

            tag_type = "ner"
            new_label_dictionary = bio_label_dictionary(corpus, args.tag_format, tag_type)

            # 4. initialize fine-tuneable transformer embeddings WITH document context
            trained_model = SequenceTagger.load(args.pretrained_model_path)
            if args.freeze_embeddings:
                trained_model.embeddings.fine_tune = False
                trained_model.embeddings.static_embeddings = True
            else:
                trained_model.embeddings.fine_tune = True
                trained_model.embeddings.static_embeddings = False

            if args.use_crf:
                if not new_label_dictionary.start_stop_tags_are_set():
                    new_label_dictionary.set_start_stop_tags()
                trained_model.use_crf = args.use_crf
                trained_model.crf = CRF(new_label_dictionary, len(new_label_dictionary), init_from_state_dict=False)
                trained_model.viterbi_decoder = ViterbiDecoder(new_label_dictionary)
                trained_model.loss_function = ViterbiLoss(new_label_dictionary)

            if args.matching_mode:
                classification_head = reuse_classification_head(
                    trained_model, args.matching_mode, new_label_dictionary, args.corpus
                )
            else:
                classification_head = random_initialized_classification_head(trained_model, new_label_dictionary)

            trained_model.tag_format = args.tag_format
            trained_model.label_dictionary = new_label_dictionary
            trained_model.tagset_size = len(new_label_dictionary) if not args.use_crf else len(new_label_dictionary) + 2
            trained_model.linear = classification_head

            # 6. initialize trainer
            if k > 0:
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
                    anneal_factor=args.anneal_factor,
                    decoder_lr_factor=args.decoder_lr_factor,
                )

                results[f"{k}"]["results"].append(result["test_score"])
            else:
                save_path = save_base_path / f"{k}shot_{seed}"
                import os

                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                result = trained_model.evaluate(corpus.test, "ner", out_path=save_path / "predictions.txt")
                results[f"{k}"]["results"].append(result.main_score)
                with open(save_path / "result.txt", "w") as f:
                    f.write(result.detailed_results)

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
    parser.add_argument("--tag_format", type=str, default="BIO")
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--pretrained_on", type=str)
    parser.add_argument("--matching_mode", type=str, default="")
    parser.add_argument("--fewnerd_granularity", type=str, default="")
    parser.add_argument("--k", type=int, default=1, nargs="+")
    parser.add_argument("--transformer", type=str, default="bert-base-uncased")
    parser.add_argument("--use_context", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mbs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--anneal_factor", type=float, default=0.5)
    parser.add_argument("--decoder_lr_factor", type=float, default=1.0)
    parser.add_argument("--use_crf", action="store_true")
    args = parser.parse_args()
    main(args)
