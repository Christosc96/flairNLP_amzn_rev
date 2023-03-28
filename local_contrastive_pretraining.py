import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import flair
from local_learn_matching_flert import generate_contrastive_pairs


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


def multiple_negatives_ranking_loss(trained_model, corpus, save_base_path) -> None:

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
