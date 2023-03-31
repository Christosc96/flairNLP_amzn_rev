import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import flair


def generate_contrastive_pairs(corpus, r_max):
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

    log_interval = 5
    lr = 2e-5
    r_max = 20
    margin = 0.5
    batch_size = 4
    max_epochs = 75
    previous_best = 10000
    patience = 3
    no_improvement = 0
    grad_norm = 0.5

    contrastive_corpus = generate_contrastive_pairs(corpus, r_max)
    dl = DataLoader(contrastive_corpus, batch_size=batch_size, collate_fn=list, shuffle=True)
    optimizer = torch.optim.Adam(trained_model.parameters(), lr=lr)
    num_batches = len(contrastive_corpus) // batch_size

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
            torch.nn.utils.clip_grad_norm_(trained_model.parameters(), grad_norm)
            optimizer.step()

            total_loss += embedding_losses.mean().item()

            if idx % log_interval == 0 and idx > 0:
                cur_loss = total_loss / log_interval
                print(
                    f"contrastive pretraining | epoch {idx:3d} | {idx:5d}/{num_batches:5d} batches | "
                    f"loss {cur_loss:2.7f}"
                )
                total_loss = 0

        print(40 * "-")
        print(
            f"contrastive pretraining | epoch {epoch} done | no improvements: {no_improvement}/{patience} | loss: {total_loss:2.7f}"
        )
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


def generate_ranking_loss_pairs(corpus, r_max):
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

    import itertools
    import random

    from flair.data import Sentence

    label2token = {_l: list(itertools.permutations(_t, 2)) for _l, _t in label2token.items()}

    token_pairs = []
    for i in range(r_max):
        _token_pairs = []
        for label, tokens in label2token.items():
            s1, s2 = random.sample(tokens, 1).pop()
            s1, s2 = Sentence(s1.text), Sentence(s2.text)
            _token_pairs.append([s1, s2])
        token_pairs.append(_token_pairs)

    return token_pairs


def multiple_negatives_ranking_pretraining(trained_model, corpus, save_base_path) -> None:

    log_interval = 5
    lr = 2e-5
    r_max = 100
    scale = 20.0
    batch_size = 4
    max_epochs = 75
    previous_best = 10000
    patience = 3
    no_improvement = 0
    grad_norm = 0.5

    ranking_loss_corpus = generate_ranking_loss_pairs(corpus, r_max)
    dl = DataLoader(ranking_loss_corpus, batch_size=batch_size, collate_fn=list, shuffle=True)
    optimizer = torch.optim.Adam(trained_model.parameters(), lr=lr)
    num_batches = len(ranking_loss_corpus) // batch_size
    ce_loss = torch.nn.CrossEntropyLoss()

    def cosine_distance(a, b):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    for epoch in range(max_epochs):
        trained_model.train()
        total_loss = 0

        for idx, batch in enumerate(dl):
            trained_model.zero_grad()
            optimizer.zero_grad()

            labels = torch.tensor(len(batch) * [range(len(batch[0]))], dtype=torch.long, device=flair.device)

            flat_batch = [
                _sample
                for _batch_item in batch
                for _batch_samples in _batch_item
                for _batch_samples in _batch_item
                for _sample in _batch_samples
            ]
            trained_model.embeddings.embed(flat_batch)

            scores = []
            for mini_batch in batch:
                anchors, positives = zip(*mini_batch)
                anchors, positives = torch.stack(
                    [token.get_embedding() for sentence in anchors for token in sentence]
                ), torch.stack([token.get_embedding() for sentence in positives for token in sentence])
                scores.append(cosine_distance(anchors, positives) * scale)

            loss = ce_loss(torch.stack(scores), labels)

            for sentence in flat_batch:
                sentence.clear_embeddings(None)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trained_model.parameters(), grad_norm)
            optimizer.step()

            total_loss += loss.item()

            if idx % log_interval == 0 and idx > 0:
                cur_loss = total_loss / log_interval
                print(
                    f"multiple negatives ranking loss pretraining | epoch {idx:3d} | {idx:5d}/{num_batches:5d} batches | "
                    f"loss {cur_loss:2.7f}"
                )
                total_loss = 0

        if (total_loss / len(dl)) < previous_best:
            previous_best = total_loss / len(dl)
            if not os.path.exists(save_base_path):
                os.mkdir(save_base_path)
            trained_model.save(save_base_path / "best-multiple-negatives-ranking-model.pt")
            no_improvement = 0
        else:
            no_improvement += 1

        print(40 * "-")
        print(
            f"multiple negatives ranking loss pretraining | epoch {epoch} done | no improvements: {no_improvement}/{patience} | loss: {total_loss:2.7f}"
        )
        print(40 * "-")

        if no_improvement > patience:
            print("stop early due to no improvments.")
            return
