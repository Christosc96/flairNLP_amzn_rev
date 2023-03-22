import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataset import Subset

import flair
from flair.data import Dictionary
from flair.models import SequenceTagger
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


def get_embeddings_per_label(trained_model, corpus):
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    sentences = [sentence for sentence in corpus.train]
    for sentence in chunks(sentences, 4):
        trained_model.embeddings.embed(sentence)
    labels_per_sentence = [[label for label in sentence.get_labels("ner")] for sentence in sentences]
    label_values_per_sentence = [
        [
            [
                "B-" + label.value if idx == 0 else "I-" + label.value
                for idx, token in enumerate(label.data_point.tokens)
            ]
            for label in sentence.get_labels("ner")
        ]
        for sentence in sentences
    ]
    embeddings_per_sentence = [
        [[token.get_embedding() for token in label.data_point.tokens] for label in labels]
        for labels in labels_per_sentence
    ]

    embeddings_per_label = {}
    for label_list, embedding_list in zip(label_values_per_sentence, embeddings_per_sentence):
        for bio_labels, embeddings in zip(label_list, embedding_list):
            for label, embedding in zip(bio_labels, embeddings):
                if label in embeddings_per_label:
                    embeddings_per_label[label].append(embedding)
                else:
                    embeddings_per_label[label] = [embedding]

    return {
        label.encode("utf-8"): torch.mean(torch.stack(embedding), dim=0)
        for label, embedding in embeddings_per_label.items()
    }


def intialize_classification_head(trained_model, corpus, tag_format: str = "BIO", tag_type: str = "ner"):

    new_label_dictionary = bio_label_dictionary(corpus, tag_format, tag_type)

    with torch.no_grad():

        embedded_labels_support_set = get_embeddings_per_label(trained_model, corpus)
        new_classification_head = torch.nn.Linear(
            trained_model.linear.in_features, len(new_label_dictionary), device=flair.device
        )

        for idx, label in enumerate(new_label_dictionary.idx2item):
            if label in embedded_labels_support_set.keys():
                similarities = torch.nn.functional.cosine_similarity(
                    embedded_labels_support_set[label], trained_model.linear.weight
                )
                similarities = similarities.clamp(min=0)
                if torch.any(similarities):
                    # scale weights to sum to 1
                    similarities = similarities / similarities.sum()
                    w = torch.mm(similarities.reshape(1, -1), trained_model.linear.weight)
                    b = torch.dot(similarities, trained_model.linear.bias)

                elif label in trained_model.label_dictionary.item2idx.keys():
                    w = trained_model.linear.weight[trained_model.label_dictionary.item2idx[label]]
                    b = trained_model.linear.weight[trained_model.label_dictionary.item2idx[label]]

                idx = new_label_dictionary.item2idx[label]
                new_classification_head.weight[idx] = w
                new_classification_head.bias[idx] = b
            elif label in trained_model.label_dictionary.item2idx.keys():
                w = trained_model.linear.weight[trained_model.label_dictionary.item2idx[label]]
                b = trained_model.linear.weight[trained_model.label_dictionary.item2idx[label]]
            else:
                pass

    trained_model.tag_format = args.tag_format
    trained_model.label_dictionary = new_label_dictionary
    trained_model.tagset_size = len(new_label_dictionary)
    trained_model.linear = new_classification_head

    return trained_model


def generate_span_pairs(corpus, r_max):
    sentences = [sentence for sentence in corpus.train]
    tokens_per_sentence = [[token for token in sentence] for sentence in corpus.train]
    idx2label_per_sentence = [
        {
            token.idx: "B-" + label.value if idx == 0 else "I-" + label.value
            for label in sentence.get_labels("ner")
            for idx, token in enumerate(label.data_point.tokens)
        }
        for sentence in sentences
    ]

    label2token = {}
    for tokens, idx2label in zip(tokens_per_sentence, idx2label_per_sentence):
        for token in tokens:
            if token.idx in idx2label:
                if idx2label.get(token.idx) not in label2token:
                    label2token[idx2label.get(token.idx)] = [token]
                else:
                    label2token[idx2label.get(token.idx)].append(token)
            else:
                if "O" not in label2token:
                    label2token["O"] = [token]
                else:
                    label2token["O"].append(token)

    import itertools
    import random

    from flair.data import Sentence

    token_pairs = []
    for label, tokens in label2token.items():
        positives = list(itertools.combinations(tokens, 2))
        if positives:
            r = min(r_max, len(positives))
            _positives = random.sample(positives, r)
            for _positive1, _positive2 in _positives:
                s1, s2 = Sentence(_positive1.text), Sentence(_positive2.text)
                token_pairs.append((1, [s1, s2]))

        for _r in range(r_max):
            possible_negatives = [key for key in label2token.keys() if not key == label]
            _negative = random.sample(possible_negatives, 1).pop()
            negative = random.sample(label2token[_negative], 1).pop()
            positive = random.sample(label2token[label], 1).pop()
            s1, s2 = Sentence(positive.text), Sentence(negative.text)
            token_pairs.append((0, [s1, s2]))

    return token_pairs


def contrastive_pretraining(trained_model, corpus, save_base_path) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    trained_model.train()
    log_interval = 5
    r_max = 20
    margin = 0.5
    token_pairs = generate_span_pairs(corpus, r_max)
    batch_size = 4
    max_epochs = 75
    previous_best = 10000
    patience = 3
    no_improvement = 0
    num_batches = len(token_pairs) // batch_size
    dl = DataLoader(token_pairs, batch_size=4, collate_fn=list, shuffle=True)
    optimizer = torch.optim.Adam(trained_model.parameters(), lr=2e-5)

    def cosine_distance(x, y):
        return 1 - F.cosine_similarity(x, y, dim=0)

    for epoch in range(max_epochs):
        trained_model.train()
        total_loss = 0
        for idx, batch in enumerate(dl):
            trained_model.zero_grad()
            optimizer.zero_grad()

            labels, data = zip(*batch)
            labels = torch.tensor(labels, device=flair.device)

            embedding_distances = []
            for mini_batch in data:
                trained_model.embeddings.embed(mini_batch)
                embedding_reps = [
                    torch.mean(torch.stack([token.get_embedding() for token in sentence]), dim=0)
                    for sentence in mini_batch
                ]
                assert len(embedding_reps) == 2
                embedding_rep_anchor, embedding_rep_other = embedding_reps
                embedding_distances.append(cosine_distance(embedding_rep_anchor, embedding_rep_other))

            embedding_distances = torch.stack(embedding_distances)
            embedding_losses = 0.5 * (
                labels.float() * embedding_distances.pow(2)
                + (1 - labels).float() * F.relu(margin - embedding_distances).pow(2)
            )

            for data_point in data:
                for s in data_point:
                    s.clear_embeddings(None)

            embedding_losses.mean().backward()
            torch.nn.utils.clip_grad_norm_(trained_model.parameters(), 0.5)
            optimizer.step()

            total_loss += embedding_losses.mean().item()

            if idx % log_interval == 0 and idx > 0:
                cur_loss = total_loss / log_interval
                print(f"| epoch {idx:3d} | {idx:5d}/{num_batches:5d} batches | " f"loss {cur_loss:2.7f}")
                total_loss = 0

        print(f"epoch {epoch} done | {total_loss / len(dl)}")
        print(40 * "-")

        if (total_loss / len(dl)) < previous_best:
            previous_best = total_loss / len(dl)
            if not os.path.exists(save_base_path):
                os.mkdir(save_base_path)
            trained_model.save(save_base_path / "best-contrastive-model.pt")
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement > patience:
            print("stop early due to no improvments.")
            return


def main(args):
    if args.cuda:
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = Path(
        f"{args.cache_path}/auto-init-fewshot-flert/"
        f"{args.transformer}{'-context' if args.use_context else ''}_"
        f"{args.corpus}{args.fewnerd_granularity}_"
        f"{'%.e' % args.lr}_{args.seed}_"
        f"{args.pretrained_on}"
        f"{'_early-stopping' if args.early_stopping else ''}"
        f"{f'_frozen-embeddings' if args.freeze_embeddings else ''}"
        f"{f'_decoder-lr-{args.lr * args.decoder_lr_factor}' if args.decoder_lr_factor != 1.0 else ''}"
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

            # 4. initialize fine-tuneable transformer embeddings WITH document context
            trained_model = SequenceTagger.load(args.pretrained_model_path)
            if args.freeze_embeddings:
                trained_model.embeddings.fine_tune = False
                trained_model.embeddings.static_embeddings = True
            else:
                trained_model.embeddings.fine_tune = True
                trained_model.embeddings.static_embeddings = False

            tag_type = "ner"

            trained_model = intialize_classification_head(
                trained_model, corpus, tag_format=args.tag_format, tag_type=tag_type
            )

            # 6. initialize trainer
            if k > 0:
                if args.contrastive_pretraining:
                    contrastive_pretraining(trained_model, corpus, save_base_path)
                    trained_model = SequenceTagger.load(save_base_path / "best-contrastive-model.pt")

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

    with open(save_base_path / "commandline_args.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)


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
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mbs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--min_lr", type=float, default=5e-7)
    parser.add_argument("--anneal_factor", type=float, default=0.5)
    parser.add_argument("--decoder_lr_factor", type=float, default=1.0)
    parser.add_argument("--contrastive_pretraining", action="store_true")
    args = parser.parse_args()
    main(args)
