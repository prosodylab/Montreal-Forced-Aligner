"""Classes for training language models"""
from __future__ import annotations

import os
import re
import subprocess
from typing import TYPE_CHECKING, Generator

from montreal_forced_aligner.abc import TopLevelMfaWorker, TrainerMixin
from montreal_forced_aligner.corpus.text_corpus import MfaWorker, TextCorpusMixin
from montreal_forced_aligner.dictionary.mixins import DictionaryMixin
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionaryMixin
from montreal_forced_aligner.models import LanguageModel

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict

__all__ = ["LmCorpusTrainer", "LmTrainerMixin", "LmArpaTrainer", "LmDictionaryCorpusTrainer"]


class LmTrainerMixin(DictionaryMixin, TrainerMixin, MfaWorker):
    """
    Abstract mixin class for training language models

    Parameters
    ----------
    prune_method: str
        Pruning method for pruning the ngram model, defaults to "relative_entropy"
    prune_thresh_small: float
        Pruning threshold for the small language model, defaults to 0.0000003
    prune_thresh_medium: float
        Pruning threshold for the medium language model, defaults to 0.0000001

    See Also
    --------
    :class:`~montreal_forced_aligner.dictionary.mixins.DictionaryMixin`
        For dictionary parsing parameters
    :class:`~montreal_forced_aligner.abc.TrainerMixin`
        For training parameters
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For worker parameters
    """

    def __init__(
        self,
        prune_method="relative_entropy",
        prune_thresh_small=0.0000003,
        prune_thresh_medium=0.0000001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prune_method = prune_method
        self.prune_thresh_small = prune_thresh_small
        self.prune_thresh_medium = prune_thresh_medium

    @property
    def mod_path(self) -> str:
        """Internal temporary path to the model file"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.mod")

    @property
    def far_path(self) -> str:
        """Internal temporary path to the FAR file"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.far")

    @property
    def large_arpa_path(self) -> str:
        """Internal temporary path to the large arpa file"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.arpa")

    @property
    def medium_arpa_path(self) -> str:
        """Internal temporary path to the medium arpa file"""
        return self.large_arpa_path.replace(".arpa", "_med.arpa")

    @property
    def small_arpa_path(self) -> str:
        """Internal temporary path to the small arpa file"""
        return self.large_arpa_path.replace(".arpa", "_small.arpa")

    def initialize_training(self) -> None:
        """Initialize training"""
        pass

    def train_iteration(self) -> None:
        """Run one training iteration"""
        pass

    def finalize_training(self) -> None:
        """Run one training iteration"""
        pass

    def prune_large_language_model(self) -> None:
        """Prune the large language model into small and medium versions"""
        self.log_info("Pruning large ngram model to medium and small versions...")
        small_mod_path = self.mod_path.replace(".mod", "_small.mod")
        med_mod_path = self.mod_path.replace(".mod", "_med.mod")
        subprocess.call(
            [
                "ngramshrink",
                f"--method={self.prune_method}",
                f"--theta={self.prune_thresh_medium}",
                self.mod_path,
                med_mod_path,
            ]
        )
        subprocess.call(["ngramprint", "--ARPA", med_mod_path, self.medium_arpa_path])

        self.log_debug("Finished pruning medium arpa!")
        subprocess.call(
            [
                "ngramshrink",
                f"--method={self.prune_method}",
                f"--theta={self.prune_thresh_small}",
                self.mod_path,
                small_mod_path,
            ]
        )
        subprocess.call(["ngramprint", "--ARPA", small_mod_path, self.small_arpa_path])

        self.log_debug("Finished pruning small arpa!")
        self.log_info("Done pruning!")

    def export_model(self, output_model_path: str) -> None:
        """
        Export language model to specified path

        Parameters
        ----------
        output_model_path:str
            Path to export model
        """
        directory, filename = os.path.split(output_model_path)
        basename, _ = os.path.splitext(filename)
        model_temp_dir = os.path.join(self.working_directory, "model_archiving")
        os.makedirs(model_temp_dir, exist_ok=True)
        model = LanguageModel.empty(basename, root_directory=model_temp_dir)
        model.add_meta_file(self)
        model.add_arpa_file(self.large_arpa_path)
        model.add_arpa_file(self.medium_arpa_path)
        model.add_arpa_file(self.small_arpa_path)
        basename, _ = os.path.splitext(output_model_path)
        model.dump(basename)


class LmCorpusTrainer(LmTrainerMixin, TextCorpusMixin, TopLevelMfaWorker):
    """
    Top-level worker to train a language model from a text corpus

    Parameters
    ----------
    order: int
        Ngram order, defaults to 3
    method:str
        Smoothing method for the ngram model, defaults to "kneser_ney"
    count_threshold:int
        Minimum count needed to not be treated as an OOV item, defaults to 1

    See Also
    --------
    :class:`~montreal_forced_aligner.language_modeling.trainer.LmTrainerMixin`
        For  language model training parsing parameters
    :class:`~montreal_forced_aligner.corpus.text_corpus.TextCorpusMixin`
        For corpus parsing parameters
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parameters
    """

    def __init__(
        self, order: int = 3, method: str = "kneser_ney", count_threshold: int = 1, **kwargs
    ):
        super().__init__(**kwargs)
        self.order = order
        self.method = method
        self.count_threshold = count_threshold

    def setup(self) -> None:
        """Set up language model training"""
        if self.initialized:
            return
        os.makedirs(self.working_log_directory, exist_ok=True)
        self._load_corpus()

        with open(self.training_path, "w", encoding="utf8") as f:
            for text in self.normalized_text_iter(self.count_threshold):
                f.write(f"{text}\n")

        self.save_oovs_found(self.working_directory)

        subprocess.call(
            ["ngramsymbols", f'--OOV_symbol="{self.oov_word}"', self.training_path, self.sym_path]
        )
        self.initialized = True

    @property
    def training_path(self):
        """Internal path to training data"""
        return os.path.join(self.working_directory, "training.txt")

    @property
    def sym_path(self):
        """Internal path to symbols file"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.sym")

    @property
    def far_path(self):
        """Internal path to FAR file"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.far")

    @property
    def cnts_path(self):
        """Internal path to counts file"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.cnts")

    @property
    def workflow_identifier(self) -> str:
        """Language model trainer identifier"""
        return "train_lm_corpus"

    @property
    def meta(self) -> MetaDict:
        """Metadata information for the language model"""
        from ..utils import get_mfa_version

        return {
            "type": "ngram",
            "order": self.order,
            "method": self.method,
            "version": get_mfa_version(),
        }

    def evaluate(self) -> None:
        """
        Run an evaluation over the training data to generate perplexity score
        """
        log_path = os.path.join(self.working_log_directory, "evaluate.log")

        small_mod_path = self.mod_path.replace(".mod", "_small.mod")
        med_mod_path = self.mod_path.replace(".mod", "_med.mod")
        with open(log_path, "w", encoding="utf8") as log_file:
            perplexity_proc = subprocess.Popen(
                [
                    "ngramperplexity",
                    f'--OOV_symbol="{self.oov_word}"',
                    self.mod_path,
                    self.far_path,
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                text=True,
            )
            stdout, stderr = perplexity_proc.communicate()
            num_sentences = None
            num_words = None
            num_oovs = None
            perplexity = None
            for line in stdout.splitlines():
                m = re.search(r"(\d+) sentences", line)
                if m:
                    num_sentences = m.group(0)
                m = re.search(r"(\d+) words", line)
                if m:
                    num_words = m.group(0)
                m = re.search(r"(\d+) OOVs", line)
                if m:
                    num_oovs = m.group(0)
                m = re.search(r"perplexity = ([\d.]+)", line)
                if m:
                    perplexity = m.group(0)

            self.log_info(f"{num_sentences} sentences, {num_words} words, {num_oovs} oovs")
            self.log_info(f"Perplexity of large model: {perplexity}")

            perplexity_proc = subprocess.Popen(
                [
                    "ngramperplexity",
                    f'--OOV_symbol="{self.oov_word}"',
                    med_mod_path,
                    self.far_path,
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                text=True,
            )
            stdout, stderr = perplexity_proc.communicate()

            perplexity = None
            for line in stdout.splitlines():
                m = re.search(r"perplexity = ([\d.]+)", line)
                if m:
                    perplexity = m.group(0)
            self.log_info(f"Perplexity of medium model: {perplexity}")
            perplexity_proc = subprocess.Popen(
                [
                    "ngramperplexity",
                    f'--OOV_symbol="{self.oov_word}"',
                    small_mod_path,
                    self.far_path,
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                text=True,
            )
            stdout, stderr = perplexity_proc.communicate()

            perplexity = None
            for line in stdout.splitlines():
                m = re.search(r"perplexity = ([\d.]+)", line)
                if m:
                    perplexity = m.group(0)
            self.log_info(f"Perplexity of small model: {perplexity}")

    def normalized_text_iter(self, min_count: int = 1) -> Generator:
        """
        Construct an iterator over the normalized texts in the corpus

        Parameters
        ----------
        min_count: int
            Minimum word count to include in the output, otherwise will use OOV code, defaults to 1

        Yields
        -------
        str
            Normalized text
        """
        unk_words = {k for k, v in self.word_counts.items() if v <= min_count}
        for u in self.utterances.values():
            text = u.text.split()
            new_text = []
            for t in text:
                if u.speaker.dictionary is not None:
                    u.speaker.dictionary.to_int(t)
                    lookup = u.speaker.dictionary.split_clitics(t)
                    if lookup is None:
                        continue
                else:
                    lookup = [t]
                for item in lookup:
                    if item in unk_words:
                        new_text.append(self.oov_word)
                        self.oovs_found[self.oov_word] += 1
                    elif (
                        u.speaker.dictionary is not None and item not in u.speaker.dictionary.words
                    ):
                        new_text.append(self.oov_word)
                    else:
                        new_text.append(item)
            yield " ".join(new_text)

    def train(self) -> None:
        """
        Train a language model
        """
        self.log_info("Beginning training large ngram model...")
        subprocess.call(
            [
                "farcompilestrings",
                "--fst_type=compact",
                f'--unknown_symbol="{self.oov_word}"',
                f"--symbols={self.sym_path}",
                "--keep_symbols",
                self.training_path,
                self.far_path,
            ]
        )
        subprocess.call(["ngramcount", f"--order={self.order}", self.far_path, self.cnts_path])
        subprocess.call(["ngrammake", f"--method={self.method}", self.cnts_path, self.mod_path])
        self.log_info("Done!")

        subprocess.call(["ngramprint", "--ARPA", self.mod_path, self.large_arpa_path])

        self.log_info("Large ngam model created!")

        self.prune_large_language_model()
        self.evaluate()


class LmDictionaryCorpusTrainer(MultispeakerDictionaryMixin, LmTrainerMixin):
    """
    Top-level worker to train a language model and incorporate a pronunciation dictionary for marking words as OOV

    See Also
    --------
    :class:`~montreal_forced_aligner.language_modeling.trainer.LmTrainerMixin`
        For language model training parsing parameters
    :class:`~montreal_forced_aligner.dictionary.multispeaker.MultispeakerDictionaryMixin`
        For dictionary parsing parameters
    """

    pass


class LmArpaTrainer(LmTrainerMixin, TopLevelMfaWorker):
    """
    Top-level worker to convert an existing ARPA-format language model to MFA format

    See Also
    --------
    :class:`~montreal_forced_aligner.language_modeling.trainer.LmTrainerMixin`
        For language model training parsing parameters
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parsing parameters
    """

    def __init__(self, arpa_path: str, **kwargs):
        super().__init__(**kwargs)
        self.arpa_path = arpa_path

    def setup(self) -> None:
        """Set up language model training"""
        self.initialized = True

    def train(self) -> None:
        """Convert the arpa model to MFA format"""
        self.log_info("Parsing large ngram model...")
        temp_text_path = os.path.join(self.working_directory, "input.arpa")
        with open(self.arpa_path, "r", encoding="utf8") as inf, open(
            temp_text_path, "w", encoding="utf8"
        ) as outf:
            for line in inf:
                outf.write(line.lower())
        subprocess.call(["ngramread", "--ARPA", temp_text_path, self.mod_path])
        os.remove(temp_text_path)

        self.log_info("Large ngam model parsed!")

        self.prune_large_language_model()
