"""
This file helps to analyze the results produced by various scripts for few-shot domain transfer.
"""


def recompute_results(path: str):
    """
    Recompute overall results file and returns it which in turn can be saved again to the target folder as results.json

    path: str = Path to the results folder that consists of typical folder structure 1shot_0, etc.
    """
    import glob

    import numpy as np

    files = glob.glob(f"{path}/*")
    results = {}
    for file in files:
        if not file.endswith(".json"):
            k = file.split("/")[-1].split("_")[0].replace("shot", "")
            with open(file + "/result.txt", "r") as f:
                for line in f.readlines():
                    if "micro avg" in line:
                        f1_score = round(float(line.split()[-2]) * 100, 2)
            if k not in results:
                results[k] = {}
                results[k]["results"] = [f1_score]
            else:
                results[k]["results"].append(f1_score)
    for key, result_dict in results.items():
        results[key]["average"] = np.mean(result_dict["results"])
        results[key]["std"] = np.std(result_dict["results"])

    return results


def scores_per_class(path: str, target_keys: list):
    """
    Computes the scores per label for all experiments in target folder. Returns a dictionary containing scores per
    class per k-shot.

    path: str = Target directory with standard result structure.
    target_keys: list = List of labels applicable for analysis. All labels can be found in local_corpora.py
    """
    import glob
    import re

    files = glob.glob(f"{path}/*")
    results = {}
    for file in files:
        if file.endswith(".json") or file.endswith(".txt") or file.endswith(".pt"):
            continue
        shot = file.split("/")[-1]
        number_of_shot = int(re.search(r"\d+", shot).group())
        if number_of_shot not in results.keys():
            results[number_of_shot] = {key: [] for key in target_keys}
        with open(file + "/result.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                for key in target_keys:
                    if key in line:
                        results[number_of_shot][key].append(line.split()[3])
    import numpy as np

    score_per_class = {}
    for key in sorted(results):
        res = results[key]
        for label, scores in res.items():
            if label not in score_per_class.keys():
                score_per_class[label] = {}

            if key not in score_per_class[label].keys():
                score_per_class[label][key] = round(np.mean([float(x) for x in scores]) * 100, 2)
    return score_per_class


def print_results():
    """
    Prints a line graph with multiple subplots with scores per class.
    Things to do before running this function:
    - Adjust figure title
    - Use correct target keys. (Below are labels for CONLL03 and WNUT17)
    - Provide description + path to target directory for plotting
    - Use applicable baselines (comment in/out in figure section)
    """
    import json

    figure_title = "Domain Transfer - CONLL03 - LPFT models with different frozen LRs - compose match"
    target_keys = ["person", "location", "organization", "miscellaneous"]
    # target_keys = ["person", "location", "corporation", "creative work", "group", "product"]

    paths = {
        "5e-1": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-5e-1_early-stopping_compose-matching",
        "1e-1": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-1_early-stopping_compose-matching",
        "5e-2": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-5e-2_early-stopping_compose-matching",
        "1e-2": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-2_early-stopping_compose-matching",
        "1e-3": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-3_early-stopping_compose-matching",
        "1e-4": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-4_early-stopping_compose-matching",
        "1e-5": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-5_early-stopping_compose-matching",
        "Linear Baseline (compose)": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/baseline_conll_03_compose-matching",
        "Dual Encoder": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/fewshot-dual-encoder/bert-base-uncased_conll_03_1e-05_123_pretrained-on-ontonotes",
    }

    def sort_dict(d):
        sorted_keys = sorted(map(int, d.keys()))
        return {k: d[str(k)] for k in sorted_keys}

    results = {}
    results_per_class = {}
    for experiment, path in paths.items():
        with open(
            f"{path}/results.json",
            "r",
        ) as f:
            result = json.load(f)
            results[experiment] = sort_dict(result)

        results_per_class[experiment] = scores_per_class(path, target_keys)

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplot_mosaic(
        [["top row", "top row"]] + [target_keys[i : i + 2] for i in range(0, len(target_keys), 2)], figsize=(12, 12)
    )

    fig.suptitle(figure_title)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))

    # BASELINES
    axes["top row"].plot(
        [1, 5, 20, 50], [44.8, 66.9, 77.5, 82.0], label="Linear (from paper)", color="tab:orange", linestyle="dotted"
    )
    axes["top row"].plot(
        [1, 5, 20, 50],
        [68.4, 76.7, 79.7, 83.1],
        label="Dual Encoder (from paper)",
        color="tab:blue",
        linestyle="dotted",
    )
    # axes["top row"].plot([1, 5, 20, 50], [27.6, 35.2, 40.9, 42.5], label="Linear (from paper)", color="tab:orange", linestyle="dotted")
    # axes["top row"].plot([1, 5, 20, 50], [38.3, 40.8, 42.7, 43.3], label="Dual Encoder (from paper)", color="tab:blue", linestyle="dotted")
    for c, (experiment, result) in zip(colors, results.items()):
        axes["top row"].plot(result.keys(), [x["average"] for x in result.values()], label=experiment, color=c)
        for label in target_keys:
            axes[label].plot(
                list(map(int, results_per_class[experiment][label].keys())),
                results_per_class[experiment][label].values(),
                label=experiment,
                color=c,
            )
    axes["top row"].set_title("Average F1 on Domain Transfer")
    axes["top row"].set_xlabel("k-shots")
    axes["top row"].set_ylabel("F1 score")
    axes["top row"].legend(loc="lower right")

    for label in target_keys:
        axes[label].set_title(f"Average F1 on label {label}")
        axes[label].set_xlabel("k-shots")
        axes[label].set_ylabel("F1 score")
        axes[label].legend(loc="lower right")

    fig.tight_layout(pad=0.9)
    fig.show()


def plot_tsne(corpus_name: str = "conll_03", matching_mode: str = "exact", model_name: str = "flert"):
    """
    Plot t-SNE plots for embeddings for either 'flert' or 'dual encoder'.
    If model_name == 'flert', you can specify whether to use 'exact' or 'compose' matching.
    """
    from local_corpora import get_corpus

    corpus = get_corpus(corpus_name, fewnerd_granularity="")

    if model_name == "flert":
        from flair.models import SequenceTagger

        pretrained_model_path = "/glusterfs/dfs-gfs-dist/goldejon/flair-models/pretrained-flert/bert-base-uncased_ontonotes_1e-05_123/final-model.pt"
        model = SequenceTagger.load(pretrained_model_path)

        from local_reuse_weights_flert import bio_label_dictionary

        new_label_dictionary = bio_label_dictionary(corpus, "BIO", "ner")

        from local_reuse_weights_flert import (
            random_initialized_classification_head,
            reuse_classification_head,
        )

        if not matching_mode == "random":
            classification_head = reuse_classification_head(model, matching_mode, new_label_dictionary, corpus_name)
        else:
            classification_head = random_initialized_classification_head(model, new_label_dictionary)

        model.tag_format = "BIO"
        model.label_dictionary = new_label_dictionary
        model.tagset_size = len(new_label_dictionary)
        model.linear = classification_head

    elif model_name == "dual encoder":
        tag_type = "ner"
        label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)
        pretrained_model_path = "/glusterfs/dfs-gfs-dist/goldejon/flair-models/pretrained-dual-encoder/bert-base-uncased_ontonotes_1e-05_123/final-model.pt"

        from flair.models import DualEncoder

        model = DualEncoder.load(pretrained_model_path)
        model._init_verbalizers_and_tag_dictionary(tag_dictionary=label_dictionary)
        new_label_dictionary = model.label_dictionary

    import torch

    with torch.no_grad():
        from flair.datasets import DataLoader, FlairDatapointDataset

        features_for_plot = []
        gold_labels_for_plot = []
        sentences = [sentence for sentence in corpus.test]
        dataloader = DataLoader(
            dataset=FlairDatapointDataset(sentences),
            batch_size=1,
        )
        for batch in dataloader:
            if model_name == "flert":
                sentence_tensor, lengths = model._prepare_tensors(batch)
                feature = model.forward(sentence_tensor, lengths)
            elif model_name == "dual encoder":
                loss, feature = model.forward(batch, inference=False, return_features=True)
            features_for_plot.append(feature.detach().cpu().numpy())
            gold_labels_for_plot.append(model._prepare_label_tensor(batch).detach().cpu().numpy())

    import numpy as np

    features_for_plot = np.concatenate(features_for_plot, axis=0)
    gold_labels_for_plot = np.concatenate(gold_labels_for_plot, axis=0)

    filtered_gold_labels_for_plot = np.array([label for label in gold_labels_for_plot if label != 0])
    filtered_features_for_plot = np.array(
        [feature for feature, label in zip(features_for_plot, gold_labels_for_plot) if label != 0]
    )

    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=2500)
    tsne_results = tsne.fit_transform(filtered_features_for_plot)

    import matplotlib.pyplot as plt

    target_ids = range(1, len(new_label_dictionary))
    labels = [label.decode("utf-8") for label in new_label_dictionary.idx2item if label != b"O"]
    plt.figure(figsize=(10, 8))
    colors = ["r", "g", "b", "c", "m", "y", "k", "orange"]
    for i, c, label in zip(target_ids, colors, labels):
        plt.scatter(
            tsne_results[filtered_gold_labels_for_plot == i, 0],
            tsne_results[filtered_gold_labels_for_plot == i, 1],
            c=c,
            label=label,
            s=1,
        )
    plt.legend(labels, labelcolor=colors, loc="lower right")
    plt.title(f"t-SNE plot - {corpus_name} {matching_mode}")
    plt.show()


def plot_tsne_labels(corpus_name: str = "conll_03", matching_mode: str = "exact", model_name: str = "flert"):
    """
    Plots the embedded labels as t-SNE. Same parameters as above.
    """
    import numpy as np

    from local_corpora import get_corpus

    corpus = get_corpus(corpus_name, fewnerd_granularity="")

    if model_name == "flert":
        from flair.models import SequenceTagger

        pretrained_model_path = "/glusterfs/dfs-gfs-dist/goldejon/flair-models/pretrained-flert/bert-base-uncased_ontonotes_1e-05_123/final-model.pt"
        model = SequenceTagger.load(pretrained_model_path)

        from local_reuse_weights_flert import bio_label_dictionary

        new_label_dictionary = bio_label_dictionary(corpus, "BIO", "ner")

        from local_reuse_weights_flert import (
            random_initialized_classification_head,
            reuse_classification_head,
        )

        if not matching_mode == "random":
            classification_head = reuse_classification_head(model, matching_mode, new_label_dictionary, corpus_name)
        else:
            classification_head = random_initialized_classification_head(model, new_label_dictionary)

    elif model_name == "dual encoder":
        tag_type = "ner"
        label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)
        pretrained_model_path = "/glusterfs/dfs-gfs-dist/goldejon/flair-models/pretrained-dual-encoder/bert-base-uncased_ontonotes_1e-05_123/final-model.pt"

        from flair.models import DualEncoder

        model = DualEncoder.load(pretrained_model_path)
        model._init_verbalizers_and_tag_dictionary(tag_dictionary=label_dictionary)
        new_label_dictionary = model.label_dictionary

    import torch

    with torch.no_grad():
        if model_name == "flert":
            features = classification_head.weight.detach().cpu().numpy()
        elif model_name == "dual encoder":
            from flair.data import Sentence

            verbalized_labels = list(map(Sentence, model.idx2verbalized_label.values()))
            model.label_encoder.embed(verbalized_labels)
            features = np.stack([label.get_embedding().detach().cpu().numpy() for label in verbalized_labels])

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(features)

    import matplotlib.pyplot as plt

    target_ids = list(range(len(new_label_dictionary)))
    labels = [label.decode("utf-8") for label in new_label_dictionary.idx2item]
    plt.figure(figsize=(8, 6))
    colors = ["r", "g", "b", "c", "m", "y", "k", "orange", "brown"]
    for i, c, label in zip(target_ids, colors, labels):
        plt.scatter(
            pca_results[np.array(target_ids) == i, 0], pca_results[np.array(target_ids) == i, 1], c=c, label=label
        )
    plt.legend(labels, labelcolor=colors, loc="lower right")
    plt.title(f"t-SNE plot for embeddings - {model_name} {corpus_name} {matching_mode}")
    plt.show()


def to_csv(save_path: str, save_path_per_class: str):
    """
    Save results as csv.

    save_path: str = The path for overall results.
    save_path_per_class = The path where to store results per class.
    """
    # target_keys = ["person", "location", "corporation", "creative work", "group", "product"]
    target_keys = ["person", "location", "organization", "miscellaneous"]

    paths = {
        "5e-1": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-5e-1_early-stopping_compose-matching",
        "1e-1": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-1_early-stopping_compose-matching",
        "5e-2": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-5e-2_early-stopping_compose-matching",
        "1e-2": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-2_early-stopping_compose-matching",
        "1e-3": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-3_early-stopping_compose-matching",
        "1e-4": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-4_early-stopping_compose-matching",
        "1e-5": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-5_early-stopping_compose-matching",
        "Linear Baseline (compose)": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/baseline_conll_03_compose-matching",
        "Dual Encoder": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/fewshot-dual-encoder/bert-base-uncased_conll_03_1e-05_123_pretrained-on-ontonotes",
    }

    def sort_dict(d):
        sorted_keys = sorted(map(int, d.keys()))
        return {k: d[str(k)] for k in sorted_keys}

    results = {}
    results_per_class = {}

    import json

    for experiment, path in paths.items():
        with open(
            f"{path}/results.json",
            "r",
        ) as f:
            result = json.load(f)
            results[experiment] = sort_dict(result)
        results_per_class[experiment] = scores_per_class(path, target_keys)

    df_data = {}
    for experiment, result_dict in results.items():
        for k, scores in result_dict.items():
            if k not in df_data:
                df_data[k] = [round(scores["average"], 2)]
            else:
                df_data[k].append(round(scores["average"], 2))

    import pandas as pd

    df = pd.DataFrame(data=df_data, index=results.keys())
    df.to_csv(save_path)

    df_data = {}
    index_tuples = []
    for experiment, result_dict in results_per_class.items():
        for label, scores in result_dict.items():
            index_tuples.append((experiment, label))
            for k, score in scores.items():
                if k not in df_data:
                    df_data[k] = [round(score, 2)]
                else:
                    df_data[k].append(round(score, 2))

    index = pd.MultiIndex.from_tuples(index_tuples, names=["Experiment", "Label"])
    df = pd.DataFrame(data=df_data, index=index)
    df.to_csv(save_path_per_class)
