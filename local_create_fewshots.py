import json
import os

from torch.utils.data.dataset import Subset

from local_corpora import get_corpus


def main():
    """
    Each function saves a json-file under data/fewshot/ in the format fewshot_datsetname.json,
    containing a list of IDs for K~2K sampling.
    """
    if not os.path.exists("data/fewshot"):
        os.mkdir("data/fewshot")
    conll()
    wnut()
    ontonotes()
    fewnerd_fine()
    fewnerd_coarse()


def conll():
    conll_indices = {}
    conll = get_corpus("conll_03", "")

    no_docstarts = []
    for idx, sentence in enumerate(conll.train):
        if "DOCSTART" in sentence.text:
            pass
        else:
            no_docstarts.append(idx)
    conll._train = Subset(conll._train, no_docstarts)

    tag_type = "ner"
    label_dict = conll.make_label_dictionary(tag_type, add_unk=False)
    labels = [label.decode("utf-8") for label in label_dict.idx2item]
    for k in [1, 2, 4, 5, 8, 10, 16, 25, 32, 50, 64]:
        for seed in range(5):
            indices = conll._sample_n_way_k_shots(
                dataset=conll._train, labels=labels, tag_type=tag_type, n=-1, k=k, seed=seed, return_indices=True
            )
            conll_indices[f"{k}-{seed}"] = indices

    with open("data/fewshot/fewshot_conll_03.json", "w") as f:
        json.dump(conll_indices, f)


def wnut():
    wnut_indices = {}
    wnut = get_corpus("wnut_17", "")

    tag_type = "ner"
    label_dict = wnut.make_label_dictionary(tag_type, add_unk=False)
    labels = [label.decode("utf-8") for label in label_dict.idx2item]
    for k in [1, 2, 4, 5, 8, 10, 16, 25, 32, 50, 64]:
        for seed in range(5):
            indices = wnut._sample_n_way_k_shots(
                dataset=wnut._train, labels=labels, tag_type=tag_type, n=-1, k=k, seed=seed, return_indices=True
            )
            wnut_indices[f"{k}-{seed}"] = indices

    with open("data/fewshot/fewshot_wnut_17.json", "w") as f:
        json.dump(wnut_indices, f)


def ontonotes():
    ontonotes_indices = {}
    ontonotes = get_corpus("ontonotes", "")

    tag_type = "ner"
    label_dict = ontonotes.make_label_dictionary(tag_type, add_unk=False)
    labels = [label.decode("utf-8") for label in label_dict.idx2item]
    for k in [1, 2, 4, 5, 8, 10, 16, 25, 32, 50, 64]:
        for seed in range(5):
            indices = ontonotes._sample_n_way_k_shots(
                dataset=ontonotes._train, labels=labels, tag_type=tag_type, n=-1, k=k, seed=seed, return_indices=True
            )
            ontonotes_indices[f"{k}-{seed}"] = indices

    with open("data/fewshot/fewshot_ontonotes.json", "w") as f:
        json.dump(ontonotes_indices, f)


def fewnerd_coarse():
    fewnerd_indices = {}
    fewnerd = get_corpus("fewnerd", "coarse")

    tag_type = "ner"
    label_dict = fewnerd.make_label_dictionary(tag_type, add_unk=False)
    labels = [label.decode("utf-8") for label in label_dict.idx2item]
    for k in [1, 2, 4, 5, 8, 10, 16, 25, 32, 50, 64]:
        for seed in range(5):
            indices = fewnerd._sample_n_way_k_shots(
                dataset=fewnerd._train, labels=labels, tag_type=tag_type, n=-1, k=k, seed=seed, return_indices=True
            )
            fewnerd_indices[f"{k}-{seed}"] = indices

    with open("data/fewshot/fewshot_fewnerdcoarse.json", "w") as f:
        json.dump(fewnerd_indices, f)


def fewnerd_fine():
    fewnerd_indices = {}
    fewnerd = get_corpus("fewnerd", "fine")

    tag_type = "ner"
    label_dict = fewnerd.make_label_dictionary(tag_type, add_unk=False)
    labels = [label.decode("utf-8") for label in label_dict.idx2item]
    for k in [1, 2, 4, 5, 8, 10, 16, 25, 32, 50, 64]:
        for seed in range(5):
            indices = fewnerd._sample_n_way_k_shots(
                dataset=fewnerd._train, labels=labels, tag_type=tag_type, n=-1, k=k, seed=seed, return_indices=True
            )
            fewnerd_indices[f"{k}-{seed}"] = indices

    with open("data/fewshot/fewshot_fewnerdfine.json", "w") as f:
        json.dump(fewnerd_indices, f)


if __name__ == "__main__":
    main()
