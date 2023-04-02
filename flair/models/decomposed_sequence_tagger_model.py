import itertools
import logging
from pathlib import Path
from typing import Any, Dict, Union

import torch
import torch.nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import flair.nn

from .sequence_tagger_model import SequenceTagger

log = logging.getLogger("flair")


class DecomposedSequenceTagger(SequenceTagger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.schema_linear = torch.nn.Linear(self.embeddings.embedding_length, len(self.tag_format), bias=False)
        raw_entities = [
            label.encode("utf-8")
            for label in list(
                dict.fromkeys([label.decode("utf-8").split("-")[-1] for label in self.label_dictionary.idx2item])
            )
        ]
        self.entity_linear = torch.nn.Linear(self.embeddings.embedding_length, len(raw_entities), bias=False)
        decomposed_mapping = itertools.product(raw_entities, self.tag_format)
        decomposed_mapping = [
            f"{schema}-{entity.decode('utf-8')}".encode("utf-8") for entity, schema in decomposed_mapping
        ]
        self.decomposed_mapping_mask = torch.tensor(
            [
                True if bio_label in self.label_dictionary.item2idx or bio_label == b"O-O" else False
                for bio_label in decomposed_mapping
            ],
            dtype=torch.bool,
        ).view(len(raw_entities), len(self.tag_format))
        assert self.decomposed_mapping_mask.sum().item() == len(self.label_dictionary)

        self.to(flair.device)

    def forward(self, sentence_tensor: torch.Tensor, lengths: torch.LongTensor):  # type: ignore[override]
        """
        Forward propagation through network.
        :param sentence_tensor: A tensor representing the batch of sentences.
        :param lengths: A IntTensor representing the lengths of the respective sentences.
        """
        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.reproject_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
            packed = pack_padded_sequence(sentence_tensor, lengths, batch_first=True)
            rnn_output, hidden = self.rnn(packed)
            sentence_tensor, output_lengths = pad_packed_sequence(rnn_output, batch_first=True)

        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        schema_features = self.schema_linear(sentence_tensor)
        entity_features = self.entity_linear(sentence_tensor)

        batch_size, seq_length, _ = schema_features.shape
        features = torch.bmm(
            entity_features.view(batch_size * seq_length, -1, 1), schema_features.view(batch_size * seq_length, 1, -1)
        )
        features = features[self.decomposed_mapping_mask.repeat(batch_size * seq_length, 1, 1)].view(
            batch_size, seq_length, len(self.label_dictionary.idx2item)
        )
        scores = self._get_scores_from_features(features, lengths)

        return scores

    @classmethod
    def load(cls, model_path: Union[str, Path, Dict[str, Any]]) -> "DecomposedSequenceTagger":
        from typing import cast

        return cast("DecomposedSequenceTagger", super().load(model_path=model_path))
