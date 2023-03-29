def main():
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer("distilbert-base-uncased")
    train_examples = [
        InputExample(texts=["Anchor 1", "Positive 1"]),
        InputExample(texts=["Anchor 2", "Positive 2"]),
        InputExample(texts=["Anchor 3", "Positive 3"]),
        InputExample(texts=["Anchor 4", "Positive 4"]),
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)


if __name__ == "__main__":
    main()
