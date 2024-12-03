"""
Model classes
=============

"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import typing
from pathlib import Path
from shutil import copy, copyfile, make_archive, move, rmtree, unpack_archive
from typing import TYPE_CHECKING, Collection, Dict, List, Optional, Tuple, Union

import pynini
import pywrapfst
import requests
import yaml
from _kalpy.gmm import AmDiagGmm
from _kalpy.hmm import TransitionModel
from _kalpy.matrix import FloatMatrix
from kalpy.feat.mfcc import MfccComputer
from kalpy.feat.pitch import PitchComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.gmm.utils import read_gmm_model
from kalpy.utils import read_kaldi_object
from rich.pretty import pprint

from montreal_forced_aligner.abc import MfaModel, ModelExporterMixin
from montreal_forced_aligner.data import Language, PhoneSetType
from montreal_forced_aligner.exceptions import (
    LanguageModelNotFoundError,
    ModelLoadError,
    ModelsConnectionError,
    PronunciationAcousticMismatchError,
    RemoteModelNotFoundError,
    RemoteModelVersionNotFoundError,
)
from montreal_forced_aligner.helper import EnhancedJSONEncoder, mfa_open

if TYPE_CHECKING:
    from dataclasses import dataclass

    from montreal_forced_aligner.abc import MetaDict
    from montreal_forced_aligner.dictionary.mixins import DictionaryMixin
    from montreal_forced_aligner.g2p.trainer import G2PTrainer
    from montreal_forced_aligner.tokenization.trainer import TokenizerTrainer
else:
    from dataclassy import dataclass

# default format for output
FORMAT = "zip"

logger = logging.getLogger("mfa")

__all__ = [
    "Archive",
    "LanguageModel",
    "AcousticModel",
    "IvectorExtractorModel",
    "DictionaryModel",
    "G2PModel",
    "ModelManager",
    "ModelRelease",
    "MODEL_TYPES",
    "guess_model_type",
]


def guess_model_type(path: Path) -> List[str]:
    """
    Guess a model type given a path

    Parameters
    ----------
    path: :class:`~pathlib.Path`
        Model archive to guess

    Returns
    -------
    list[str]
        Possible model types that use that extension
    """
    ext = os.path.splitext(path)[1]
    if not ext:
        return []
    possible = []
    for m, mc in MODEL_TYPES.items():
        if ext in mc.extensions:
            possible.append(m)
    return possible


class Archive(MfaModel):
    """
    Class representing data in a directory or archive file (zip, tar,
    tar.gz/tgz)

    Based on the prosodylab-aligner
    (https://github.com/prosodylab/Prosodylab-Aligner) archive class.

    Parameters
    ----------
    source: :class:`~pathlib.Path`
        Source path
    root_directory: :class:`~pathlib.Path`
        Root directory to unpack and store temporary files
    """

    extensions = [".zip"]

    model_type = None

    def __init__(
        self,
        source: typing.Union[str, Path],
        root_directory: Optional[typing.Union[str, Path]] = None,
    ):
        from .config import get_temporary_directory

        if isinstance(source, str):
            source = Path(source)
        source = source.resolve()
        if root_directory is None:
            root_directory = get_temporary_directory().joinpath(
                "extracted_models", self.model_type
            )
        if isinstance(root_directory, str):
            root_directory = Path(root_directory)
        self.root_directory = root_directory
        self.source = source
        self._meta = {}
        self.name = source.stem
        if os.path.isdir(source):
            self.dirname = source
        else:
            self.dirname = root_directory.joinpath(f"{self.name}_{self.model_type}")
            if self.dirname.exists():
                shutil.rmtree(self.dirname, ignore_errors=True)

            os.makedirs(root_directory, exist_ok=True)
            unpack_archive(source, self.dirname)
            files = [x for x in self.dirname.iterdir()]
            old_dir_path = files[0]
            if len(files) == 1 and old_dir_path.is_dir():  # Backwards compatibility
                for f in old_dir_path.iterdir():
                    f = f.relative_to(old_dir_path)
                    move(old_dir_path.joinpath(f), self.dirname.joinpath(f))
                old_dir_path.rmdir()

    def parse_old_features(self) -> None:
        """
        Parse MFA model's features and ensure that they are up-to-date with current functionality
        """
        if "features" not in self._meta:
            return
        feature_key_remapping = {
            "type": "feature_type",
            "deltas": "uses_deltas",
            "fmllr": "uses_speaker_adaptation",
        }

        for key, new_key in feature_key_remapping.items():
            if key in self._meta["features"]:
                self._meta["features"][new_key] = self._meta["features"][key]
                del self._meta["features"][key]
        if "uses_splices" not in self._meta["features"]:  # Backwards compatibility
            self._meta["features"]["uses_splices"] = os.path.exists(
                self.dirname.joinpath("lda.mat")
            )
        if "uses_speaker_adaptation" not in self._meta["features"]:
            self._meta["features"]["uses_speaker_adaptation"] = os.path.exists(
                self.dirname.joinpath("final.alimdl")
            )

    def get_subclass_object(
        self,
    ) -> Union[AcousticModel, G2PModel, LanguageModel, TokenizerModel, IvectorExtractorModel]:
        """
        Instantiate subclass models based on files contained in the archive

        Returns
        -------
        :class:`~montreal_forced_aligner.models.AcousticModel`, :class:`~montreal_forced_aligner.models.G2PModel`, :class:`~montreal_forced_aligner.models.LanguageModel`, or :class:`~montreal_forced_aligner.models.IvectorExtractorModel`
            Subclass model that was auto detected

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.ModelLoadError`
            If the model type cannot be determined
        """
        files = [x.name for x in self.dirname.iterdir()]

        if "tree" in files:
            return AcousticModel(self.dirname, self.root_directory)
        if "phones.sym" in files or "phones.txt" in files:
            return G2PModel(self.dirname, self.root_directory)
        if any(f.endswith(".arpa") for f in files):
            return LanguageModel(self.dirname, self.root_directory)
        if "final.ie" in files:
            return IvectorExtractorModel(self.dirname, self.root_directory)
        if "tokenizer.fst" in files:
            return TokenizerModel(self.dirname, self.root_directory)
        raise ModelLoadError(self.source)

    @classmethod
    def valid_extension(cls, filename: Path) -> bool:
        """
        Check whether a file has a valid extension for the given model archive

        Parameters
        ----------
        filename: :class:`~pathlib.Path`
            File name to check

        Returns
        -------
        bool
            True if the extension matches the models allowed extensions
        """
        if filename.suffix in cls.extensions:
            return True
        return False

    @classmethod
    def generate_path(
        cls, root: Path, name: str, enforce_existence: bool = True
    ) -> Optional[Path]:
        """
        Generate a path for a given model from the root directory and the name of the model

        Parameters
        ----------
        root: :class:`~pathlib.Path`
            Root directory for the full path
        name: str
            Name of the model
        enforce_existence: bool
            Flag to return None if the path doesn't exist, defaults to True

        Returns
        -------
        Path
           Full path in the root directory for the model
        """
        for ext in cls.extensions:
            path = root.joinpath(name + ext)
            if path.exists() or not enforce_existence:
                return path
        return None

    def pretty_print(self) -> None:
        """
        Pretty print the archive's meta data using rich
        """
        pprint({"Archive": {"name": self.name, "data": self.meta}})

    @property
    def meta(self) -> dict:
        """
        Get the meta data associated with the model
        """
        if not self._meta:
            meta_path = self.dirname.joinpath("meta.json")
            format = "json"
            if not os.path.exists(meta_path):
                meta_path = self.dirname.joinpath("meta.yaml")
                format = "yaml"
            with mfa_open(meta_path, "r") as f:
                if format == "yaml":
                    self._meta = yaml.load(f, Loader=yaml.Loader)
                else:
                    self._meta = json.load(f)
        self.parse_old_features()
        return self._meta

    def add_meta_file(self, trainer: ModelExporterMixin) -> None:
        """
        Add a metadata file from a given trainer to the model

        Parameters
        ----------
        trainer: :class:`~montreal_forced_aligner.abc.ModelExporterMixin`
            The trainer to construct the metadata from
        """
        with mfa_open(self.dirname.joinpath("meta.json"), "w") as f:
            json.dump(trainer.meta, f, ensure_ascii=False)

    @classmethod
    def empty(
        cls, head: str, root_directory: Optional[typing.Union[str, Path]] = None
    ) -> Union[
        Archive, IvectorExtractorModel, AcousticModel, G2PModel, TokenizerModel, LanguageModel
    ]:
        """
        Initialize an archive using an empty directory

        Parameters
        ----------
        head: str
            Directory name to create
        root_directory: str, optional
            Root directory to create temporary data, defaults to the MFA temporary directory

        Returns
        -------
        :class:`~montreal_forced_aligner.models.Archive`, :class:`~montreal_forced_aligner.models.AcousticModel`, :class:`~montreal_forced_aligner.models.G2PModel`, :class:`~montreal_forced_aligner.models.LanguageModel`, :class:`~montreal_forced_aligner.models.TokenizerModel`, or :class:`~montreal_forced_aligner.models.IvectorExtractorModel`
            Model constructed from the empty directory
        """
        from .config import get_temporary_directory

        if root_directory is None:
            root_directory = get_temporary_directory().joinpath("temp_models", cls.model_type)

        source = root_directory.joinpath(head)
        source.mkdir(parents=True, exist_ok=True)
        return cls(source, root_directory)

    def add(self, source: str):
        """
        Add file into archive

        Parameters
        ----------
        source: str
            Path to file to copy into the directory
        """
        copy(source, self.dirname)

    def __repr__(self) -> str:
        """Representation string of a model"""
        return f"{self.__class__.__name__}(dirname={self.dirname!r})"

    def clean_up(self) -> None:
        """Remove temporary directory"""
        rmtree(self.dirname)

    def dump(self, path: Path, archive_fmt: str = FORMAT) -> str:
        """
        Write archive to disk, and return the name of final archive

        Parameters
        ----------
        path: :class:`~pathlib.Path`
            Path to write to
        archive_fmt: str, optional
            Archive extension to use, defaults to ".zip"

        Returns
        -------
        str
            Path of constructed archive
        """
        return make_archive(os.path.splitext(path)[0], archive_fmt, *os.path.split(self.dirname))


class AcousticModel(Archive):
    """
    Class for storing acoustic models in MFA, exported as zip files containing the necessary Kaldi files
    to be reused

    """

    files = [
        "final.mdl",
        "final.alimdl",
        "lda.mat",
        "phone_pdf.counts",
        # "rules.yaml",
        "tokenizer.fst",
        "phone_lm.fst",
        "tree",
        "rules.yaml",
        "phones.txt",
        "graphemes.txt",
    ]
    extensions = [".zip", ".am"]

    model_type = "acoustic"

    def __init__(
        self,
        source: typing.Union[str, Path],
        root_directory: Optional[typing.Union[str, Path]] = None,
    ):
        if source in AcousticModel.get_available_models():
            source = AcousticModel.get_pretrained_path(source)

        super().__init__(source, root_directory)
        self._am = None
        self._tm = None

    @property
    def version(self):
        return self.meta["version"]

    @property
    def uses_cmvn(self):
        return self.meta["features"]["uses_cmvn"]

    @property
    def language(self) -> Language:
        return Language[self.meta.get("language", "unknown")]

    def add_meta_file(self, trainer: ModelExporterMixin) -> None:
        """
        Add metadata file from a model trainer

        Parameters
        ----------
        trainer: :class:`~montreal_forced_aligner.abc.ModelExporterMixin`
            Trainer to supply metadata information about the acoustic model
        """
        with mfa_open(self.dirname.joinpath("meta.json"), "w") as f:
            json.dump(trainer.meta, f, ensure_ascii=False)

    @property
    def parameters(self) -> MetaDict:
        """Parameters to pass to top-level workers"""
        params = {**self.meta["features"]}
        params["non_silence_phones"] = {x for x in self.meta["phones"]}
        params["oov_phone"] = self.meta["oov_phone"]
        params["language"] = self.meta["language"]
        params["optional_silence_phone"] = self.meta["optional_silence_phone"]
        params["phone_set_type"] = self.meta["phone_set_type"]
        params["silence_probability"] = self.meta.get("silence_probability", 0.5)
        params["initial_silence_probability"] = self.meta.get("initial_silence_probability", 0.5)
        params["final_non_silence_correction"] = self.meta.get(
            "final_non_silence_correction", None
        )
        params["final_silence_correction"] = self.meta.get("final_silence_correction", None)
        if "other_noise_phone" in self.meta:
            params["other_noise_phone"] = self.meta["other_noise_phone"]
        if (
            "dictionaries" in self.meta
            and "position_dependent_phones" in self.meta["dictionaries"]
        ):
            params["position_dependent_phones"] = self.meta["dictionaries"][
                "position_dependent_phones"
            ]
        else:
            params["position_dependent_phones"] = self.meta.get("position_dependent_phones", True)
        return params

    @property
    def tree_path(self) -> Path:
        """Current acoustic model path"""
        return self.dirname.joinpath("tree")

    @property
    def lda_mat_path(self) -> Path:
        """Current acoustic model path"""
        return self.dirname.joinpath("lda.mat")

    @property
    def model_path(self) -> Path:
        """Current acoustic model path"""
        return self.dirname.joinpath("final.mdl")

    @property
    def phone_symbol_path(self) -> Path:
        """Path to phone symbol table"""
        return self.dirname.joinpath("phones.txt")

    @property
    def rules_path(self) -> Path:
        """Path to phone symbol table"""
        return self.dirname.joinpath("rules.yaml")

    @property
    def alignment_model_path(self) -> Path:
        """Alignment model path"""
        path = self.model_path.with_suffix(".alimdl")
        if os.path.exists(path):
            return path
        return self.model_path

    @property
    def acoustic_model(self) -> AmDiagGmm:
        if self._am is None:
            self._tm, self._am = read_gmm_model(self.alignment_model_path)
        return self._am

    @property
    def transition_model(self) -> TransitionModel:
        if self._tm is None:
            self._tm, self._am = read_gmm_model(self.alignment_model_path)
        return self._tm

    @property
    def lexicon_compiler(self):
        lc = LexiconCompiler(
            silence_probability=self.meta.get("silence_probability", 0.5),
            initial_silence_probability=self.meta.get("initial_silence_probability", 0.5),
            final_silence_correction=self.meta.get("final_silence_correction", None),
            final_non_silence_correction=self.meta.get("final_non_silence_correction", None),
            silence_phone=self.meta.get("optional_silence_phone", "sil"),
            oov_phone=self.meta.get("oov_phone", "sil"),
            position_dependent_phones=self.meta.get("position_dependent_phones", False),
            phones={x for x in self.meta["phones"]},
        )
        if self.meta.get("phone_mapping", None):
            lc.phone_table = pywrapfst.SymbolTable()
            for k, v in self.meta["phone_mapping"].items():
                lc.phone_table.add_symbol(k, v)
        elif self.phone_symbol_path.exists():
            lc.phone_table = pywrapfst.SymbolTable.read_text(self.phone_symbol_path)
        return lc

    @property
    def mfcc_computer(self) -> MfccComputer:
        return MfccComputer(**self.mfcc_options)

    @property
    def pitch_computer(self) -> typing.Optional[PitchComputer]:
        if self.meta["features"].get("use_pitch", False):
            return PitchComputer(**self.pitch_options)
        return

    @property
    def lda_mat(self) -> FloatMatrix:
        lda_mat_path = self.dirname.joinpath("lda.mat")
        lda_mat = None
        if lda_mat_path.exists():
            lda_mat = read_kaldi_object(FloatMatrix, lda_mat_path)
        return lda_mat

    @property
    def mfcc_options(self) -> MetaDict:
        """Parameters to use in computing MFCC features."""
        return {
            "sample_frequency": self.meta["features"].get("sample_frequency", 16000),
            "frame_shift": self.meta["features"].get("frame_shift", 10),
            "frame_length": self.meta["features"].get("frame_length", 25),
            "dither": self.meta["features"].get("dither", 0.0001),
            "preemphasis_coefficient": self.meta["features"].get("preemphasis_coefficient", 0.97),
            "snip_edges": self.meta["features"].get("snip_edges", True),
            "num_mel_bins": self.meta["features"].get("num_mel_bins", 23),
            "low_frequency": self.meta["features"].get("low_frequency", 20),
            "high_frequency": self.meta["features"].get("high_frequency", 7800),
            "num_coefficients": self.meta["features"].get("num_coefficients", 13),
            "use_energy": self.meta["features"].get("use_energy", False),
            "energy_floor": self.meta["features"].get("energy_floor", 1.0),
            "raw_energy": self.meta["features"].get("raw_energy", True),
            "cepstral_lifter": self.meta["features"].get("cepstral_lifter", 22),
        }

    @property
    def pitch_options(self) -> MetaDict:
        """Parameters to use in computing MFCC features."""
        use_pitch = self.meta["features"].get("use_pitch", False)
        use_voicing = self.meta["features"].get("use_voicing", False)
        use_delta_pitch = self.meta["features"].get("use_delta_pitch", False)
        normalize = self.meta["features"].get("normalize_pitch", True)
        options = {
            "frame_shift": self.meta["features"].get("frame_shift", 10),
            "frame_length": self.meta["features"].get("frame_length", 25),
            "min_f0": self.meta["features"].get("min_f0", 50),
            "max_f0": self.meta["features"].get("max_f0", 800),
            "sample_frequency": self.meta["features"].get("sample_frequency", 16000),
            "penalty_factor": self.meta["features"].get("penalty_factor", 0.1),
            "delta_pitch": self.meta["features"].get("delta_pitch", 0.005),
            "snip_edges": self.meta["features"].get("snip_edges", True),
            "add_normalized_log_pitch": False,
            "add_delta_pitch": False,
            "add_pov_feature": False,
        }
        if use_pitch:
            options["add_normalized_log_pitch"] = normalize
            options["add_raw_log_pitch"] = not normalize
        options["add_delta_pitch"] = use_delta_pitch
        options["add_pov_feature"] = use_voicing
        return options

    @property
    def lda_options(self) -> MetaDict:
        """Parameters to use in computing MFCC features."""
        return {
            "splice_left_context": self.meta["features"].get("splice_left_context", 3),
            "splice_right_context": self.meta["features"].get("splice_right_context", 3),
        }

    @property
    def meta(self) -> MetaDict:
        """
        Metadata information for the acoustic model
        """
        default_features = {
            "feature_type": "mfcc",
            "use_energy": False,
            "frame_shift": 10,
            "snip_edges": True,
            "low_frequency": 20,
            "high_frequency": 7800,
            "sample_frequency": 16000,
            "allow_downsample": True,
            "allow_upsample": True,
            "use_pitch": False,
            "use_voicing": False,
            "uses_cmvn": True,
            "uses_deltas": True,
            "uses_splices": False,
            "uses_voiced": False,
            "uses_speaker_adaptation": False,
            "silence_weight": 0.0,
            "fmllr_update_type": "full",
            "splice_left_context": 3,
            "splice_right_context": 3,
        }
        if not self._meta:
            meta_path = self.dirname.joinpath("meta.json")
            format = "json"
            if not os.path.exists(meta_path):
                meta_path = self.dirname.joinpath("meta.yaml")
                format = "yaml"
            if not os.path.exists(meta_path):
                self._meta = {
                    "version": "0.9.0",
                    "architecture": "gmm-hmm",
                    "features": default_features,
                }
            else:
                with mfa_open(meta_path, "r") as f:
                    if format == "yaml":
                        self._meta = yaml.load(f, Loader=yaml.Loader)
                    else:
                        self._meta = json.load(f)
                if self._meta["features"] == "mfcc+deltas":
                    self._meta["features"] = default_features
                    if "pitch" in self._meta["features"]:
                        self._meta["features"]["use_pitch"] = self._meta["features"].pop("pitch")
                if (
                    self._meta["features"].get("use_pitch", False)
                    and self._meta["version"] < "2.0.6"
                ):
                    self._meta["features"]["use_delta_pitch"] = True
            if "phone_type" not in self._meta:
                self._meta["phone_type"] = "triphone"
            if "optional_silence_phone" not in self._meta:
                self._meta["optional_silence_phone"] = "sil"
            if "oov_phone" not in self._meta:
                self._meta["oov_phone"] = "spn"
            if format == "yaml":
                self._meta["other_noise_phone"] = "sp"
            if "phone_set_type" not in self._meta:
                self._meta["phone_set_type"] = "UNKNOWN"
            if "language" not in self._meta or self._meta["version"] < "3.0":
                self._meta["language"] = "unknown"
            self._meta["phones"] = set(self._meta.get("phones", []))
            if (
                "uses_speaker_adaptation" not in self._meta["features"]
                or not self._meta["features"]["uses_speaker_adaptation"]
            ):
                self._meta["features"]["uses_speaker_adaptation"] = os.path.exists(
                    self.dirname.joinpath("final.alimdl")
                )
            if self._meta["version"] in {"0.9.0", "1.0.0"}:
                self._meta["features"]["uses_speaker_adaptation"] = True
            if (
                "uses_splices" not in self._meta["features"]
                or not self._meta["features"]["uses_splices"]
            ):
                self._meta["features"]["uses_splices"] = os.path.exists(
                    self.dirname.joinpath("lda.mat")
                )
                if self._meta["features"]["uses_splices"]:
                    self._meta["features"]["uses_deltas"] = False
            if (
                self._meta["features"].get("use_pitch", False)
                and "use_voicing" not in self._meta["features"]
            ):
                self._meta["features"]["use_voicing"] = True
            if (
                "dictionaries" in self._meta
                and "position_dependent_phones" not in self._meta["dictionaries"]
            ):
                if self._meta["version"] < "2.0":
                    default_value = True
                else:
                    default_value = False
                self._meta["dictionaries"]["position_dependent_phones"] = self._meta.get(
                    "position_dependent_phones", default_value
                )
        self.parse_old_features()
        return self._meta

    def pretty_print(self) -> None:
        """
        Prints the metadata information to the terminal
        """

        configuration_data = {"Acoustic model": {"name": self.name, "data": {}}}
        configuration_data["Acoustic model"]["data"]["Version"] = (self.meta["version"],)

        if "citation" in self.meta:
            configuration_data["Acoustic model"]["data"]["Citation"] = self.meta["citation"]
        if "train_date" in self.meta:
            configuration_data["Acoustic model"]["data"]["Train date"] = self.meta["train_date"]
        configuration_data["Acoustic model"]["data"]["Architecture"] = self.meta["architecture"]
        configuration_data["Acoustic model"]["data"]["Phone type"] = self.meta["phone_type"]
        configuration_data["Acoustic model"]["data"]["Features"] = {
            "Feature type": self.meta["features"]["feature_type"],
            "Frame shift": self.meta["features"]["frame_shift"],
            "Performs speaker adaptation": self.meta["features"]["uses_speaker_adaptation"],
            "Performs LDA on features": self.meta["features"]["uses_splices"],
        }
        if self.meta["phones"]:
            configuration_data["Acoustic model"]["data"]["Phones"] = self.meta["phones"]
        else:
            configuration_data["Acoustic model"]["data"]["Phones"] = "None found!"

        pprint(configuration_data)

    def add_model(self, source: Path) -> None:
        """
        Add file into archive

        Parameters
        ----------
        source: str
            File to add
        """
        for f in self.files:
            source_path = source.joinpath(f)
            dest_path = self.dirname.joinpath(f)
            if source_path.exists():
                if f == "phones.txt":
                    with mfa_open(source_path, "r") as in_f, mfa_open(dest_path, "w") as out_f:
                        for line in in_f:
                            if re.match(r"#\d+", line):
                                continue
                            out_f.write(line)
                else:
                    copyfile(source_path, dest_path)

    def add_pronunciation_models(
        self, source: Path, dictionary_base_names: Collection[str]
    ) -> None:
        """
        Add file into archive

        Parameters
        ----------
        source: str
            File to add
        dictionary_base_names: list[str]
            Base names of dictionaries to add pronunciation models
        """
        for base_name in dictionary_base_names:
            for f in [f"{base_name}.fst", f"{base_name}_align.fst"]:
                if source.joinpath(f).exists():
                    copyfile(source.joinpath(f), self.dirname.joinpath(f))

    def export_model(self, destination: Path) -> None:
        """
        Extract the model files to a new directory

        Parameters
        ----------
        destination: str
            Destination directory to extract files to
        """
        destination.mkdir(parents=True, exist_ok=True)
        for f in self.files:
            if os.path.exists(self.dirname.joinpath(f)):
                copyfile(self.dirname.joinpath(f), destination.joinpath(f))

    def log_details(self) -> None:
        """
        Log metadata information to a logger
        """
        logger.debug("")
        logger.debug("====ACOUSTIC MODEL INFO====")
        logger.debug("Acoustic model root directory: " + str(self.root_directory))
        logger.debug("Acoustic model dirname: " + str(self.dirname))
        meta_path = self.dirname.joinpath("meta.json")
        if not os.path.exists(meta_path):
            meta_path = self.dirname.joinpath("meta.yaml")
        logger.debug("Acoustic model meta path: " + str(meta_path))
        if not os.path.exists(meta_path):
            logger.debug("META.YAML DOES NOT EXIST, this may cause issues in validating the model")
        logger.debug("Acoustic model meta information:")
        stream = yaml.dump(self.meta)
        logger.debug(stream)
        logger.debug("")

    def validate(self, dictionary: DictionaryMixin) -> None:
        """
        Validate this acoustic model against a pronunciation dictionary to ensure their
        phone sets are compatible

        Parameters
        ----------
        dictionary: :class:`~montreal_forced_aligner.dictionary.mixins.DictionaryMixin`
            DictionaryMixin  to compare phone sets with

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.PronunciationAcousticMismatchError`
            If there are phones missing from the acoustic model
        """
        if isinstance(dictionary, G2PModel):
            missing_phones = dictionary.meta["phones"] - set(self.meta["phones"])
        else:
            missing_phones = dictionary.non_silence_phones - set(self.meta["phones"])
        missing_phones -= {"sp", "<eps>"}
        if missing_phones:  # Compatibility
            raise (PronunciationAcousticMismatchError(missing_phones))


class IvectorExtractorModel(Archive):
    """
    Model class for IvectorExtractor
    """

    model_type = "ivector"

    model_files = [
        "final.ie",
        "final.ubm",
        "final.dubm",
        "ivector_lda.mat",
        "plda",
        "num_utts.ark",
        "speaker_ivectors.ark",
    ]
    extensions = [
        ".ivector",
        ".zip",
    ]

    def __init__(
        self,
        source: typing.Union[str, Path],
        root_directory: Optional[typing.Union[str, Path]] = None,
    ):
        if source in IvectorExtractorModel.get_available_models():
            source = IvectorExtractorModel.get_pretrained_path(source)

        super().__init__(source, root_directory)

    @property
    def parameters(self) -> MetaDict:
        """Parameters to pass to top-level workers"""
        params = {**self.meta["features"]}
        for key in ["ivector_dimension", "num_gselect", "min_post", "posterior_scale"]:
            params[key] = self.meta[key]
        return params

    def add_model(self, source: str) -> None:
        """
        Add file into archive

        Parameters
        ----------
        source: str
            File to add
        """
        for filename in self.model_files:
            if os.path.exists(os.path.join(source, filename)):
                copyfile(os.path.join(source, filename), self.dirname.joinpath(filename))

    def export_model(self, destination: str) -> None:
        """
        Extract the model files to a new directory

        Parameters
        ----------
        destination: str
            Destination directory to extract files to
        """
        os.makedirs(destination, exist_ok=True)
        for filename in self.model_files:
            if os.path.exists(self.dirname.joinpath(filename)):
                copyfile(self.dirname.joinpath(filename), os.path.join(destination, filename))

    @property
    def mfcc_options(self) -> MetaDict:
        """Parameters to use in computing MFCC features."""
        return {
            "use_energy": self._meta["features"].get("use_energy", False),
            "dither": self._meta["features"].get("dither", 0.0001),
            "energy_floor": self._meta["features"].get("energy_floor", 1.0),
            "num_coefficients": self._meta["features"].get("num_coefficients", 13),
            "num_mel_bins": self._meta["features"].get("num_mel_bins", 23),
            "cepstral_lifter": self._meta["features"].get("cepstral_lifter", 22),
            "preemphasis_coefficient": self._meta["features"].get("preemphasis_coefficient", 0.97),
            "frame_shift": self._meta["features"].get("frame_shift", 10),
            "frame_length": self._meta["features"].get("frame_length", 25),
            "low_frequency": self._meta["features"].get("low_frequency", 20),
            "high_frequency": self._meta["features"].get("high_frequency", 7800),
            "sample_frequency": self._meta["features"].get("sample_frequency", 16000),
            "snip_edges": self._meta["features"].get("snip_edges", True),
        }

    @property
    def pitch_options(self) -> MetaDict:
        """Parameters to use in computing MFCC features."""
        use_pitch = self._meta["features"].get("use_pitch", False)
        use_voicing = self._meta["features"].get("use_voicing", False)
        use_delta_pitch = self._meta["features"].get("use_delta_pitch", False)
        normalize = self._meta["features"].get("normalize_pitch", True)
        options = {
            "frame_shift": self._meta["features"].get("frame_shift", 10),
            "frame_length": self._meta["features"].get("frame_length", 25),
            "min_f0": self._meta["features"].get("min_f0", 50),
            "max_f0": self._meta["features"].get("max_f0", 800),
            "sample_frequency": self._meta["features"].get("sample_frequency", 16000),
            "penalty_factor": self._meta["features"].get("penalty_factor", 0.1),
            "delta_pitch": self._meta["features"].get("delta_pitch", 0.005),
            "snip_edges": self._meta["features"].get("snip_edges", True),
            "add_normalized_log_pitch": False,
            "add_delta_pitch": False,
            "add_pov_feature": False,
        }
        if use_pitch:
            options["add_normalized_log_pitch"] = normalize
            options["add_raw_log_pitch"] = not normalize
        if self._meta["version"] == "2.1.0" and "ivector_dimension" in self._meta:
            options["add_normalized_log_pitch"] = True
            options["add_raw_log_pitch"] = True
        options["add_delta_pitch"] = use_delta_pitch
        options["add_pov_feature"] = use_voicing
        return options


class G2PModel(Archive):
    """
    Class for G2P models

    Parameters
    ----------
    source: str
        Path to source archive
    root_directory: str
        Path to save exported model
    """

    extensions = [".zip", ".g2p"]

    model_type = "g2p"

    def __init__(
        self,
        source: typing.Union[str, Path],
        root_directory: Optional[typing.Union[str, Path]] = None,
    ):
        if source in G2PModel.get_available_models():
            source = G2PModel.get_pretrained_path(source)

        super().__init__(source, root_directory)

    @property
    def fst(self):
        return pynini.Fst.read(self.fst_path)

    @property
    def phone_table(self):
        return pywrapfst.SymbolTable.read_text(self.sym_path)

    @property
    def grapheme_table(self):
        return pywrapfst.SymbolTable.read_text(self.grapheme_sym_path)

    @property
    def rewriter(self):
        if not self.grapheme_sym_path.exists():
            return None
        if self.meta["architecture"] == "phonetisaurus":
            from montreal_forced_aligner.g2p.generator import PhonetisaurusRewriter

            rewriter = PhonetisaurusRewriter(
                self.fst,
                self.grapheme_table,
                self.phone_table,
                num_pronunciations=1,
                grapheme_order=self.meta["grapheme_order"],
                graphemes=self.meta["graphemes"],
                sequence_separator=self.meta["sequence_separator"],
                strict=True,
                unicode_decomposition=self.meta["unicode_decomposition"],
            )
        else:
            from montreal_forced_aligner.g2p.generator import Rewriter

            rewriter = Rewriter(
                self.fst,
                self.grapheme_table,
                self.phone_table,
                num_pronunciations=1,
                strict=True,
                unicode_decomposition=self.meta["unicode_decomposition"],
            )
        return rewriter

    def add_meta_file(self, g2p_trainer: G2PTrainer) -> None:
        """
        Construct metadata information for the G2P model from the dictionary it was trained from

        Parameters
        ----------
        g2p_trainer: :class:`~montreal_forced_aligner.g2p.trainer.G2PTrainer`
            Trainer for the G2P model
        """

        with mfa_open(self.dirname.joinpath("meta.json"), "w") as f:
            json.dump(g2p_trainer.meta, f, cls=EnhancedJSONEncoder)

    @property
    def meta(self) -> dict:
        """Metadata for the G2P model"""
        if not self._meta:
            meta_path = self.dirname.joinpath("meta.json")
            format = "json"
            if not os.path.exists(meta_path):
                meta_path = self.dirname.joinpath("meta.yaml")
                format = "yaml"
            if not os.path.exists(meta_path):
                self._meta = {"version": "0.9.0", "architecture": "phonetisaurus"}
            else:
                with mfa_open(meta_path, "r") as f:
                    if format == "json":
                        self._meta = json.load(f)
                    else:
                        self._meta = yaml.load(f, Loader=yaml.Loader)
            self._meta["phones"] = set(self._meta.get("phones", []))
            self._meta["graphemes"] = set(self._meta.get("graphemes", []))
            self._meta["evaluation"] = self._meta.get("evaluation", [])
            self._meta["training"] = self._meta.get("training", [])
            self._meta["unicode_decomposition"] = self._meta.get("unicode_decomposition", False)
        return self._meta

    @property
    def fst_path(self) -> Path:
        """G2P model's FST path"""
        return self.dirname.joinpath("model.fst")

    @property
    def sym_path(self) -> Path:
        """G2P model's symbols path"""
        path = self.dirname.joinpath("phones.txt")
        if path.exists():
            return path
        return self.dirname.joinpath("phones.sym")

    @property
    def grapheme_sym_path(self) -> Path:
        """G2P model's grapheme symbols path"""
        path = self.dirname.joinpath("graphemes.txt")
        if path.exists():
            return path
        return self.dirname.joinpath("graphemes.sym")

    def add_sym_path(self, source_directory: Path) -> None:
        """
        Add symbols file into archive

        Parameters
        ----------
        source_directory: str
            Source directory path
        """
        if not os.path.exists(self.sym_path):
            copyfile(os.path.join(source_directory, "phones.txt"), self.sym_path)
        if not os.path.exists(self.grapheme_sym_path) and os.path.exists(
            os.path.join(source_directory, "graphemes.txt")
        ):
            copyfile(os.path.join(source_directory, "graphemes.txt"), self.grapheme_sym_path)

    def add_fst_model(self, source_directory: Path) -> None:
        """
        Add FST file into archive

        Parameters
        ----------
        source_directory: str
            Source directory path
        """
        if not self.fst_path.exists():
            copyfile(os.path.join(source_directory, "model.fst"), self.fst_path)

    def export_fst_model(self, destination: str) -> None:
        """
        Extract FST model path to destination

        Parameters
        ----------
        destination: str
            Destination directory
        """
        os.makedirs(destination, exist_ok=True)
        copy(self.fst_path, destination)

    def validate(self, word_list: Collection[str]) -> bool:
        """
        Validate the G2P model against a word list to ensure that all graphemes are known

        Parameters
        ----------
        word_list: Collection[str]
            Word list to validate against

        Returns
        -------
        bool
            False if missing graphemes were found
        """
        graphemes = set()
        for w in word_list:
            graphemes.update(w)
        missing_graphemes = graphemes - self.meta["graphemes"]
        if missing_graphemes:
            return False
        else:
            return True


class TokenizerModel(Archive):
    """
    Class for Tokenizer models

    Parameters
    ----------
    source: str
        Path to source archive
    root_directory: str
        Path to save exported model
    """

    extensions = [".zip", ".tkn"]

    model_type = "tokenizer"

    def __init__(
        self,
        source: typing.Union[str, Path],
        root_directory: Optional[typing.Union[str, Path]] = None,
    ):
        if source in TokenizerModel.get_available_models():
            source = TokenizerModel.get_pretrained_path(source)

        super().__init__(source, root_directory)

    def add_meta_file(self, g2p_trainer: TokenizerTrainer) -> None:
        """
        Construct metadata information for the G2P model from the dictionary it was trained from

        Parameters
        ----------
        g2p_trainer: :class:`~montreal_forced_aligner.g2p.trainer.G2PTrainer`
            Trainer for the G2P model
        """

        with mfa_open(self.dirname.joinpath("meta.json"), "w") as f:
            json.dump(g2p_trainer.meta, f, cls=EnhancedJSONEncoder)

    @property
    def meta(self) -> dict:
        """Metadata for the G2P model"""
        if not self._meta:
            meta_path = self.dirname.joinpath("meta.json")
            format = "json"
            if not os.path.exists(meta_path):
                meta_path = self.dirname.joinpath("meta.yaml")
                format = "yaml"
            if not os.path.exists(meta_path):
                self._meta = {"version": "0.9.0", "architecture": "pynini"}
            else:
                with mfa_open(meta_path, "r") as f:
                    if format == "json":
                        self._meta = json.load(f)
                    else:
                        self._meta = yaml.load(f, Loader=yaml.Loader)
            self._meta["evaluation"] = self._meta.get("evaluation", [])
            self._meta["training"] = self._meta.get("training", [])
        return self._meta

    @property
    def fst_path(self) -> Path:
        """Tokenizer model's FST path"""
        return self.dirname.joinpath("tokenizer.fst")

    @property
    def sym_path(self) -> Path:
        """Tokenizer model's grapheme symbols path"""
        path = self.dirname.joinpath("graphemes.txt")
        if path.exists():
            return path
        path = self.dirname.joinpath("graphemes.sym")
        if path.exists():
            return path
        return self.dirname.joinpath("graphemes.syms")

    @property
    def input_sym_path(self) -> Path:
        """Tokenizer model's input symbols path"""
        path = self.dirname.joinpath("input.txt")
        if path.exists():
            return path
        return self.dirname.joinpath("input.syms")

    @property
    def output_sym_path(self) -> Path:
        """Tokenizer model's output symbols path"""
        path = self.dirname.joinpath("output.txt")
        if path.exists():
            return path
        return self.dirname.joinpath("output.syms")

    def add_graphemes_path(self, source_directory: Path) -> None:
        """
        Add symbols file into archive

        Parameters
        ----------
        source_directory: :class:`~pathlib.Path`
            Source directory path
        """
        for p in [self.sym_path, self.output_sym_path, self.input_sym_path]:
            source_p = source_directory.joinpath(p.name)
            if not p.exists() and source_p.exists():
                copyfile(source_p, p)

    def add_tokenizer_model(self, source_directory: Path) -> None:
        """
        Add FST file into archive

        Parameters
        ----------
        source_directory: :class:`~pathlib.Path`
            Source directory path
        """
        if not self.fst_path.exists():
            copyfile(source_directory.joinpath("tokenizer.fst"), self.fst_path)

    def export_fst_model(self, destination: Path) -> None:
        """
        Extract FST model path to destination

        Parameters
        ----------
        destination: :class:`~pathlib.Path`
            Destination directory
        """
        destination.mkdir(parents=True, exist_ok=True)
        copy(self.fst_path, destination)

    def validate(self, *args) -> None:
        """
        Placeholder
        """
        pass


class LanguageModel(Archive):
    """
    Class for MFA language models

    Parameters
    ----------
    source: str
        Path to source archive
    root_directory: str
        Path to save exported model
    """

    model_type = "language_model"

    arpa_extension = ".arpa"
    extensions = [f".{FORMAT}", arpa_extension, ".lm"]

    def __init__(
        self,
        source: typing.Union[str, Path],
        root_directory: Optional[typing.Union[str, Path]] = None,
    ):
        if source in LanguageModel.get_available_models():
            source = LanguageModel.get_pretrained_path(source)
        from .config import get_temporary_directory

        if isinstance(source, str):
            source = Path(source)
        if root_directory is None:
            root_directory = get_temporary_directory().joinpath(
                "extracted_models", self.model_type
            )
        if isinstance(root_directory, str):
            source = Path(root_directory)

        if source.suffix == self.arpa_extension:
            self.root_directory = root_directory
            self._meta = {}
            self.name = source.stem
            self.dirname = root_directory.joinpath(f"{self.name}_{self.model_type}")
            if not os.path.exists(self.dirname):
                os.makedirs(self.dirname, exist_ok=True)
            copy(source, self.large_arpa_path)
        else:
            super().__init__(source, root_directory)

    @property
    def decode_arpa_path(self) -> Path:
        """
        Uses the smallest language model for decoding
        """
        for path in [self.small_arpa_path, self.medium_arpa_path, self.large_arpa_path]:
            if path.exists():
                return path
        raise LanguageModelNotFoundError(self.small_arpa_path)

    @property
    def carpa_path(self) -> Path:
        """
        Uses the largest language model for rescoring
        """
        for path in [self.large_arpa_path, self.medium_arpa_path, self.small_arpa_path]:
            if path.exists():
                return path
        raise LanguageModelNotFoundError(self.large_arpa_path)

    @property
    def small_arpa_path(self) -> Path:
        """Small arpa path"""
        for path in self.dirname.iterdir():
            if path.name.endswith("_small" + self.arpa_extension):
                return path
        return self.dirname.joinpath(f"{self.name}_small{self.arpa_extension}")

    @property
    def medium_arpa_path(self) -> Path:
        """Medium arpa path"""
        for path in self.dirname.iterdir():
            if path.name.endswith("_med" + self.arpa_extension):
                return path
        return self.dirname.joinpath(f"{self.name}_med{self.arpa_extension}")

    @property
    def large_arpa_path(self) -> Path:
        """Large arpa path"""
        for path in self.dirname.iterdir():
            if (
                path.name.endswith(self.arpa_extension)
                and "_small" not in path.name
                and "_med" not in path.name
            ):
                return path
        return self.dirname.joinpath(self.name + self.arpa_extension)

    def add_arpa_file(self, arpa_path: Path) -> None:
        """
        Adds an ARPA file to the model

        Parameters
        ----------
        arpa_path: :class:`~pathlib.Path`
            Path to ARPA file
        """
        output_name = self.large_arpa_path
        if arpa_path.name.endswith("_small.arpa"):
            output_name = self.small_arpa_path
        elif arpa_path.name.endswith("_medium.arpa"):
            output_name = self.medium_arpa_path
        copyfile(arpa_path, output_name)


class DictionaryModel(MfaModel):
    """
    Class for representing MFA pronunciation dictionaries

    Parameters
    ----------
    path: :class:`~pathlib.Path`
        Path to the dictionary file
    root_directory: :class:`~pathlib.Path`, optional
        Path to working directory (currently not needed, but present to maintain consistency with other MFA Models
    """

    model_type = "dictionary"

    extensions = [".dict", ".txt", ".yaml", ".yml"]

    def __init__(
        self,
        path: typing.Union[str, Path],
        root_directory: Optional[typing.Union[str, Path]] = None,
        phone_set_type: typing.Union[str, PhoneSetType] = "UNKNOWN",
    ):
        if path in DictionaryModel.get_available_models():
            path = DictionaryModel.get_pretrained_path(path)
        if isinstance(path, str):
            path = Path(path)
        if root_directory is None:
            from montreal_forced_aligner.config import get_temporary_directory

            root_directory = get_temporary_directory().joinpath(
                "extracted_models", self.model_type
            )
        if isinstance(root_directory, str):
            root_directory = Path(root_directory)
        self.path = path
        self.dirname = root_directory.joinpath(f"{self.name}_{self.model_type}")
        self.pronunciation_probabilities = True
        self.silence_probabilities = True
        self.oov_probabilities = True
        if not isinstance(phone_set_type, PhoneSetType):
            phone_set_type = PhoneSetType[phone_set_type]
        self.phone_set_type = phone_set_type
        detect_phone_set = False
        if self.phone_set_type == PhoneSetType.AUTO:
            detect_phone_set = True

        patterns = {
            PhoneSetType.ARPA: PhoneSetType.ARPA.regex_detect,
            PhoneSetType.IPA: PhoneSetType.IPA.regex_detect,
            PhoneSetType.PINYIN: PhoneSetType.PINYIN.regex_detect,
        }
        counts = {
            PhoneSetType.UNKNOWN: 0,
            PhoneSetType.ARPA: 0,
            PhoneSetType.IPA: 0,
            PhoneSetType.PINYIN: 0,
        }

        count = 0
        with mfa_open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if detect_phone_set:
                    for phone_set, pattern in patterns.items():
                        if pattern.search(line):
                            counts[phone_set] += 1

                            break
                    else:
                        counts[PhoneSetType.UNKNOWN] += 1
                        continue
                    if counts[phone_set] > 100:
                        other_sets_max = max(counts[x] for x in counts if x is not phone_set)
                        if counts[phone_set] - other_sets_max >= 100:
                            break
                else:
                    count += 1
                    if count > 15:
                        break
                _, line = line.split(maxsplit=1)  # word
                try:
                    next_item, line = line.split(maxsplit=1)
                except ValueError:
                    next_item = line
                    line = ""
                if self.pronunciation_probabilities:
                    try:
                        prob = float(next_item)
                        if prob > 1 or prob < 0:
                            raise ValueError
                    except ValueError:
                        self.pronunciation_probabilities = False
                try:
                    next_item, line = line.split(maxsplit=1)
                except ValueError:
                    self.silence_probabilities = False
                    continue
                if self.silence_probabilities:
                    try:
                        prob = float(next_item)
                        if prob > 1 or prob < 0:
                            raise ValueError
                    except ValueError:
                        self.silence_probabilities = False
                self.oov_probabilities = False
        if detect_phone_set:
            self.phone_set_type = max(counts.keys(), key=lambda x: counts[x])

    @property
    def meta(self) -> MetaDict:
        """Metadata for the dictionary"""
        return {
            "phone_set_type": self.phone_set_type,
            "pronunciation_probabilities": self.pronunciation_probabilities,
            "silence_probabilities": self.silence_probabilities,
        }

    def add_meta_file(self, trainer: ModelExporterMixin) -> None:
        """Not implemented method"""
        raise NotImplementedError

    def pretty_print(self) -> None:
        """
        Pretty print the dictionary's metadata
        """
        from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionary

        configuration_data = {"Dictionary": {"name": self.name, "data": self.meta}}
        temp_directory = self.dirname.joinpath("temp")
        if temp_directory.exists():
            shutil.rmtree(temp_directory)
        dictionary = MultispeakerDictionary(self.path, phone_set_type=self.phone_set_type)
        graphemes, phone_counts = dictionary.dictionary_setup()
        configuration_data["Dictionary"]["data"]["phones"] = sorted(dictionary.non_silence_phones)
        configuration_data["Dictionary"]["data"]["detailed_phone_info"] = {}
        if self.phone_set_type.has_base_phone_regex:
            for k, v in sorted(dictionary.base_phones.items()):
                if k not in configuration_data["Dictionary"]["data"]["detailed_phone_info"]:
                    configuration_data["Dictionary"]["data"]["detailed_phone_info"][k] = []
                for p2 in sorted(v, key=lambda x: -phone_counts[x]):
                    detail_string = f"{p2} ({phone_counts[p2]})"
                    configuration_data["Dictionary"]["data"]["detailed_phone_info"][k].append(
                        detail_string
                    )

        else:
            configuration_data["Dictionary"]["data"]["detailed_phone_info"] = {}
            for phone in sorted(dictionary.non_silence_phones):
                configuration_data["Dictionary"]["data"]["detailed_phone_info"][
                    phone
                ] = phone_counts[phone]
        if len(graphemes) < 50:
            configuration_data["Dictionary"]["data"]["graphemes"] = sorted(graphemes)
        else:
            configuration_data["Dictionary"]["data"]["graphemes"] = f"{len(graphemes)} graphemes"
        pprint(configuration_data)

    @classmethod
    def valid_extension(cls, filename: Path) -> bool:
        """
        Check whether a file has a valid extension for the given model archive

        Parameters
        ----------
        filename: :class:`~pathlib.Path`
            File name to check

        Returns
        -------
        bool
            True if the extension matches the models allowed extensions
        """
        if filename.suffix in cls.extensions:
            return True
        return False

    @classmethod
    def generate_path(
        cls, root: Path, name: str, enforce_existence: bool = True
    ) -> Optional[Path]:
        """
        Generate a path for a given model from the root directory and the name of the model

        Parameters
        ----------
        root: :class:`~pathlib.Path`
            Root directory for the full path
        name: str
            Name of the model
        enforce_existence: bool
            Flag to return None if the path doesn't exist, defaults to True

        Returns
        -------
        Path
           Full path in the root directory for the model
        """
        for ext in cls.extensions:
            path = root.joinpath(name + ext)
            if path.exists() or not enforce_existence:
                return path
        return None

    @property
    def is_multiple(self) -> bool:
        """Flag for whether the dictionary contains multiple lexicons"""
        return self.path.suffix in [".yaml", ".yml"]

    @property
    def name(self) -> str:
        """Name of the dictionary"""
        return self.path.stem

    def load_dictionary_paths(self) -> Dict[str, Tuple[DictionaryModel, typing.Set[str]]]:
        """
        Load the pronunciation dictionaries

        Returns
        -------
        dict[str, :class:`~montreal_forced_aligner.models.DictionaryModel`]
            Mapping of component pronunciation dictionaries
        """
        mapping = {}
        if self.is_multiple:
            with mfa_open(self.path, "r") as f:
                data = yaml.load(f, Loader=yaml.Loader)
                for speaker, path in data.items():
                    if path not in mapping:
                        if path != "nonnative":
                            path = DictionaryModel(path)
                        mapping[path] = (path, set())
                    mapping[path][1].add(speaker)
        else:
            mapping[str(self.path)] = (self, {"default"})
        return mapping


MODEL_TYPES = {
    "acoustic": AcousticModel,
    "g2p": G2PModel,
    "dictionary": DictionaryModel,
    "language_model": LanguageModel,
    "ivector": IvectorExtractorModel,
    "tokenizer": TokenizerModel,
}


@dataclass(slots=True)
class ModelRelease:
    """
    Dataclas for model releases

    Parameters
    ----------
    model_name: str
        Name of the model
    tag_name: str
        Tag on GitHub
    version: str
        Version of the model
    download_link: str
        Link to download the model
    download_file_name: str
        File name to save as
    release_id: int
        Release ID on GitHub
    """

    model_name: str
    tag_name: str
    version: str
    download_link: str
    download_file_name: str
    release_id: int = None

    @property
    def release_link(self) -> Optional[str]:
        """Generate link pointing to the release on GitHub"""
        if not self.release_id:
            return None
        return ModelManager.base_url + f"/{self.release_id}"


class ModelManager:
    """
    Class for managing the currently available models on the local system and the models available to be downloaded

    Parameters
    ----------
        token: str, optional
            GitHub authentication token to use to increase release limits
        hf_token: str, optional
            HuggingFace authentication token to use to increase release limits
        ignore_cache: bool
            Flag to ignore previously downloaded files
    """

    base_url = "https://api.github.com/repos/MontrealCorpusTools/mfa-models/releases"

    def __init__(
        self,
        token: typing.Optional[str] = None,
        hf_token: typing.Optional[str] = None,
        ignore_cache: bool = False,
    ):
        from montreal_forced_aligner.config import get_temporary_directory

        pretrained_dir = get_temporary_directory().joinpath("pretrained_models")
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        self.local_models = {k: [] for k in MODEL_TYPES.keys()}
        self.remote_models: Dict[str, Dict[str, Dict[str, ModelRelease]]] = {
            k: {} for k in MODEL_TYPES.keys()
        }
        self.token = token
        environment_token = os.environ.get("GITHUB_TOKEN", None)
        if self.token is None:
            self.token = environment_token
        self.hf_token = hf_token
        environment_token = os.environ.get("HF_TOKEN", None)
        if self.hf_token is None:
            self.hf_token = environment_token
        self.synced_remote = False
        self.ignore_cache = ignore_cache
        self._cache_info = {}
        self.refresh_local()

    @property
    def cache_path(self) -> Path:
        """Path to json file with cached etags and download links"""
        from montreal_forced_aligner.config import get_temporary_directory

        return get_temporary_directory().joinpath("pretrained_models", "cache.json")

    def reset_local(self) -> None:
        """Reset cached models"""
        from montreal_forced_aligner.config import get_temporary_directory

        pretrained_dir = get_temporary_directory().joinpath("pretrained_models")
        if pretrained_dir.exists():
            shutil.rmtree(pretrained_dir, ignore_errors=True)

    def refresh_local(self) -> None:
        """Refresh cached information with the latest list of local model"""
        if self.cache_path.exists() and not self.ignore_cache:
            reset_cache = False
            with mfa_open(self.cache_path, "r") as f:
                self._cache_info = json.load(f)
                for (
                    model_type,
                    model_releases,
                ) in self._cache_info.items():  # Backward compatibility
                    if model_type not in MODEL_TYPES:
                        continue
                    for version_data in model_releases.values():
                        if not isinstance(version_data, dict):
                            reset_cache = True
                            break
                    if reset_cache:
                        break
                if reset_cache:
                    self._cache_info = {}
                if "list_etags" in self._cache_info:
                    self._cache_info["list_etags"] = {
                        int(k): v for k, v in self._cache_info["list_etags"].items()
                    }
        self.local_models = {
            model_type: model_class.get_available_models()
            for model_type, model_class in MODEL_TYPES.items()
        }

    def refresh_remote(self) -> None:
        """Refresh cached information with the latest list of downloadable models"""
        self.remote_models = {k: {} for k in MODEL_TYPES.keys()}
        data_count = 100
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        else:
            logger.debug("No Github Token supplied")
        page = 1
        etags = {}
        if "list_etags" in self._cache_info:
            etags = self._cache_info["list_etags"]
        else:
            self._cache_info["list_etags"] = {}
        while data_count == 100:
            if page in etags:
                headers["If-None-Match"] = etags[page]
            r = requests.get(
                self.base_url, params={"per_page": 100, "page": page}, headers=headers
            )
            if r.status_code >= 400:
                raise ModelsConnectionError(r.status_code, r.json(), r.headers)
            if r.status_code >= 300:  # Using cached releases
                for model_type, model_releases in self._cache_info.items():
                    if model_type not in MODEL_TYPES:
                        continue
                    for model_name, version_data in model_releases.items():
                        for version, data in version_data.items():
                            if model_name not in self.remote_models[model_type]:
                                self.remote_models[model_type][model_name] = {}
                            self.remote_models[model_type][model_name][version] = ModelRelease(
                                *data
                            )
                return
            self._cache_info["list_etags"][page] = r.headers["etag"]
            data = r.json()
            data_count = len(data)
            for d in data:
                tag = d["tag_name"]
                model_type, model_name, version = tag.split(
                    "-"
                )  # tag format "{model_type}-{model_name}-v{version}"
                if model_type not in self.remote_models:  # Other releases, archived, etc
                    continue
                if not tag.startswith(model_type):
                    continue
                if "archive" in tag:
                    continue
                download_url = d["assets"][0]["url"]
                file_name = d["assets"][0]["name"]
                if model_name not in self.remote_models[model_type]:
                    self.remote_models[model_type][model_name] = {}
                self.remote_models[model_type][model_name][version] = ModelRelease(
                    model_name, tag, version, download_url, file_name, d["id"]
                )
                if model_type not in self._cache_info:
                    self._cache_info[model_type] = {}
                if model_name not in self._cache_info[model_type]:
                    self._cache_info[model_type][model_name] = {}
                self._cache_info[model_type][model_name][version] = [
                    model_name,
                    tag,
                    version,
                    download_url,
                    file_name,
                    d["id"],
                ]
            page += 1
        with mfa_open(self.cache_path, "w") as f:
            json.dump(self._cache_info, f, ensure_ascii=False)
        self.synced_remote = True

    def has_local_model(self, model_type: str, model_name: str) -> bool:
        """Check for local model"""
        return model_name in self.local_models[model_type]

    def print_local_models(self, model_type: typing.Optional[str] = None) -> None:
        """
        List all local pretrained models

        Parameters
        ----------
        model_type: str, optional
            Model type, will list models of all model types if None
        """
        self.refresh_local()
        if model_type is None:
            logger.info("Available local models")
            data = {}
            for model_type, model_class in MODEL_TYPES.items():
                data[model_type] = model_class.get_available_models()
            pprint(data)
        else:
            logger.info(f"Available local {model_type} models")
            model_class = MODEL_TYPES[model_type]
            names = model_class.get_available_models()
            if names:
                pprint(names)
            else:
                logger.error("No models found")

    def print_remote_models(self, model_type: typing.Optional[str] = None) -> None:
        """
        Print of models available for download

        Parameters
        ----------
        model_type: str
            Model type to look up
        """
        if not self.synced_remote:
            self.refresh_remote()
        if model_type is None:
            logger.info("Available models for download")
            data = {}
            for model_type, release_data in self.remote_models.items():
                data[model_type] = sorted(release_data.keys())
            pprint(data)
        else:
            logger.info(f"Available {model_type} models for download")
            names = {
                x: sorted(self.remote_models[model_type][x].keys())
                for x in sorted(self.remote_models[model_type].keys())
            }
            if names:
                pprint(names)
            else:
                logger.error("No models found")

    def download_model(
        self,
        model_type: str,
        model_name: typing.Optional[str],
        version: typing.Optional[str] = None,
    ) -> None:
        """
        Download a model to MFA's temporary directory

        Parameters
        ----------
        model_type: str
            Model type
        model_name: str
            Name of model
        version: str, optional
            Version of model to download, optional
        """
        if not model_name:
            return self.print_remote_models(model_type)
        if not self.synced_remote:
            self.refresh_remote()
        ignore_cache = self.ignore_cache
        if model_name not in self.remote_models[model_type]:
            raise RemoteModelNotFoundError(
                model_name, model_type, sorted(self.remote_models[model_type].keys())
            )
        if version is None:
            version = sorted(self.remote_models[model_type][model_name].keys())[-1]
        else:
            if not version.startswith("v"):
                version = f"v{version}"
            ignore_cache = True

        if version not in self.remote_models[model_type][model_name]:
            raise RemoteModelVersionNotFoundError(
                model_name,
                model_type,
                version,
                sorted(self.remote_models[model_type][model_name].keys()),
            )
        release = self.remote_models[model_type][model_name][version]
        local_path = (
            MODEL_TYPES[model_type].pretrained_directory().joinpath(release.download_file_name)
        )
        if local_path.exists() and not ignore_cache:
            logger.warning(
                f"Local version of model already exists ({local_path}). "
                f"Use the --ignore_cache flag to force redownloading."
            )
            return
        headers = {"Accept": "application/octet-stream"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        if release.download_link in self._cache_info:
            headers["If-None-Match"] = self._cache_info[release.download_link]
        r = requests.get(release.download_link, headers=headers)
        if r.status_code >= 400:
            raise ModelsConnectionError(r.status_code, r.json(), r.headers)
        self._cache_info[release.download_link] = r.headers["etag"]
        with mfa_open(local_path, "wb") as f:
            f.write(r.content)
        self.refresh_local()
        logger.info(
            f"Saved model to {local_path}, you can now use {model_name} in place of {model_type} paths in mfa commands."
        )
