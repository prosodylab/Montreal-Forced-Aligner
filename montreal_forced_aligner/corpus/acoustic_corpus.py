"""Class definitions for corpora"""
from __future__ import annotations

import logging
import os
import threading
import time
import typing
from abc import ABCMeta
from multiprocessing.pool import ThreadPool
from pathlib import Path
from queue import Empty, Queue
from typing import List, Optional

import sqlalchemy
from kalpy.data import KaldiMapping
from kalpy.feat.cmvn import CmvnComputer
from kalpy.feat.data import FeatureArchive
from kalpy.utils import kalpy_logger
from tqdm.rich import tqdm

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import MfaWorker
from montreal_forced_aligner.corpus.base import CorpusMixin
from montreal_forced_aligner.corpus.classes import FileData
from montreal_forced_aligner.corpus.features import (
    CalcFmllrArguments,
    CalcFmllrFunction,
    ComputeVadFunction,
    FeatureConfigMixin,
    FinalFeatureArguments,
    FinalFeatureFunction,
    MfccArguments,
    MfccFunction,
    VadArguments,
)
from montreal_forced_aligner.corpus.helper import find_exts
from montreal_forced_aligner.corpus.multiprocessing import (
    AcousticDirectoryParser,
    CorpusProcessWorker,
)
from montreal_forced_aligner.data import DatabaseImportData, PhoneType, WordType, WorkflowType
from montreal_forced_aligner.db import (
    Corpus,
    CorpusWorkflow,
    File,
    Phone,
    PhoneInterval,
    SoundFile,
    Speaker,
    TextFile,
    Utterance,
    Word,
    bulk_update,
)
from montreal_forced_aligner.dictionary.mixins import DictionaryMixin
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionaryMixin
from montreal_forced_aligner.exceptions import (
    CorpusError,
    FeatureGenerationError,
    SoundFileError,
    TextGridParseError,
    TextParseError,
)
from montreal_forced_aligner.helper import load_scp, mfa_open
from montreal_forced_aligner.textgrid import parse_aligned_textgrid
from montreal_forced_aligner.utils import Counter, run_kaldi_function

__all__ = [
    "AcousticCorpusMixin",
    "AcousticCorpus",
    "AcousticCorpusWithPronunciations",
    "AcousticCorpusPronunciationMixin",
]

logger = logging.getLogger("mfa")


class AcousticCorpusMixin(CorpusMixin, FeatureConfigMixin, metaclass=ABCMeta):
    """
    Mixin class for acoustic corpora

    Parameters
    ----------
    audio_directory: str
        Extra directory to look for audio files

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.base.CorpusMixin`
        For corpus parsing parameters
    :class:`~montreal_forced_aligner.corpus.features.FeatureConfigMixin`
        For feature generation parameters

    Attributes
    ----------
    sound_file_errors: list[str]
        List of sound files with errors in loading
    stopped: :class:`~threading.Event`
        Stop check for loading the corpus
    """

    def __init__(self, audio_directory: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.audio_directory = audio_directory
        self.sound_file_errors = []
        self.stopped = threading.Event()
        self.features_generated = False
        self.transcription_done = False
        self.alignment_evaluation_done = False
        self.transcriptions_required = True

    def has_alignments(self, workflow_id: typing.Optional[int] = None) -> bool:
        with self.session() as session:
            if workflow_id is None:
                check = session.query(PhoneInterval).limit(1).first() is not None
            else:
                if isinstance(workflow_id, int):
                    check = (
                        session.query(CorpusWorkflow.alignments_collected)
                        .filter(CorpusWorkflow.id == workflow_id)
                        .scalar()
                    )
                else:
                    check = (
                        session.query(CorpusWorkflow.alignments_collected)
                        .filter(CorpusWorkflow.workflow_type == workflow_id)
                        .scalar()
                    )
        return check

    def has_ivectors(self) -> bool:
        with self.session() as session:
            check = (
                session.query(Corpus)
                .filter(Corpus.ivectors_calculated == True)  # noqa
                .limit(1)
                .first()
                is not None
            )
        return check

    def has_xvectors(self) -> bool:
        with self.session() as session:
            check = (
                session.query(Corpus)
                .filter(Corpus.xvectors_loaded == True)  # noqa
                .limit(1)
                .first()
                is not None
            )
        return check

    def has_any_ivectors(self) -> bool:
        with self.session() as session:
            check = (
                session.query(Corpus)
                .filter(
                    sqlalchemy.or_(
                        Corpus.ivectors_calculated == True, Corpus.xvectors_loaded == True  # noqa
                    )
                )
                .limit(1)
                .first()
                is not None
            )
        return check

    @property
    def no_transcription_files(self) -> List[str]:
        """List of sound files without text files"""
        with self.session() as session:
            files = session.query(SoundFile.sound_file_path).filter(
                ~sqlalchemy.exists().where(SoundFile.file_id == TextFile.file_id)
            )
            return [x[0] for x in files]

    @property
    def transcriptions_without_wavs(self) -> List[str]:
        """List of text files without sound files"""
        with self.session() as session:
            files = session.query(TextFile.text_file_path).filter(
                ~sqlalchemy.exists().where(SoundFile.file_id == TextFile.file_id)
            )
            return [x[0] for x in files]

    def inspect_database(self) -> None:
        """Check if a database file exists and create the necessary metadata"""
        self.initialize_database()
        with self.session() as session:
            corpus = session.query(Corpus).first()
            if corpus:
                self.imported = corpus.imported
                self.features_generated = corpus.features_generated
                self.text_normalized = corpus.text_normalized
            else:
                session.add(
                    Corpus(
                        name=self.data_source_identifier,
                        path=self.corpus_directory,
                        data_directory=self.corpus_output_directory,
                    )
                )
                session.commit()

    def load_reference_alignments(self, reference_directory: Path) -> None:
        """
        Load reference alignments to use in alignment evaluation from a directory

        Parameters
        ----------
        reference_directory: :class:`~pathlib.Path`
            Directory containing reference alignments

        """
        self.create_new_current_workflow(WorkflowType.reference)
        workflow = self.current_workflow
        if workflow.alignments_collected:
            logger.info("Reference alignments already loaded!")
            return
        logger.info("Loading reference files...")
        indices = []
        jobs = []
        reference_intervals = []
        with tqdm(total=self.num_files, disable=config.QUIET) as pbar, self.session() as session:
            phone_mapping = {}
            max_id = 0
            interval_id = session.query(sqlalchemy.func.max(PhoneInterval.id)).scalar()
            if not interval_id:
                interval_id = 0
            interval_id += 1
            for p, p_id in session.query(Phone.phone, Phone.id):
                phone_mapping[p] = p_id
                if p_id > max_id:
                    max_id = p_id
            new_phones = []
            for root, _, files in os.walk(reference_directory, followlinks=True):
                if root.startswith("."):  # Ignore hidden directories
                    continue
                root_speaker = os.path.basename(root)
                for f in files:
                    if f.startswith("."):  # Ignore hidden files
                        continue
                    if f.endswith(".TextGrid"):
                        file_name = f.replace(".TextGrid", "")
                        file_id = session.query(File.id).filter_by(name=file_name).scalar()
                        if not file_id:
                            continue
                        if config.USE_MP:
                            indices.append(file_id)
                            jobs.append((os.path.join(root, f), root_speaker))
                        else:
                            intervals = parse_aligned_textgrid(os.path.join(root, f), root_speaker)
                            utterances = (
                                session.query(
                                    Utterance.id, Speaker.name, Utterance.begin, Utterance.end
                                )
                                .join(Utterance.speaker)
                                .filter(Utterance.file_id == file_id)
                                .order_by(Utterance.begin)
                            )
                            for u_id, speaker_name, begin, end in utterances:
                                if speaker_name not in intervals:
                                    continue
                                while intervals[speaker_name]:
                                    interval = intervals[speaker_name].pop(0)
                                    dur = interval.end - interval.begin
                                    mid_point = interval.begin + (dur / 2)
                                    if begin <= mid_point <= end:
                                        if interval.label not in phone_mapping:
                                            max_id += 1
                                            phone_mapping[interval.label] = max_id
                                            new_phones.append(
                                                {
                                                    "id": max_id,
                                                    "mapping_id": max_id - 1,
                                                    "phone": interval.label,
                                                    "kaldi_label": interval.label,
                                                    "phone_type": PhoneType.extra,
                                                }
                                            )
                                        reference_intervals.append(
                                            {
                                                "id": interval_id,
                                                "begin": interval.begin,
                                                "end": interval.end,
                                                "phone_id": phone_mapping[interval.label],
                                                "utterance_id": u_id,
                                                "workflow_id": workflow.id,
                                            }
                                        )
                                        interval_id += 1
                                    if mid_point > end:
                                        intervals[speaker_name].insert(0, interval)
                                        break

                            pbar.update(1)

            if config.USE_MP:
                with ThreadPool(config.NUM_JOBS) as pool:
                    gen = pool.starmap(parse_aligned_textgrid, jobs)
                    for i, intervals in enumerate(gen):
                        pbar.update(1)
                        file_id = indices[i]
                        utterances = (
                            session.query(
                                Utterance.id, Speaker.name, Utterance.begin, Utterance.end
                            )
                            .join(Utterance.speaker)
                            .filter(Utterance.file_id == file_id)
                            .order_by(Utterance.begin)
                        )
                        for u_id, speaker_name, begin, end in utterances:
                            if speaker_name not in intervals:
                                continue
                            while intervals[speaker_name]:
                                interval = intervals[speaker_name].pop(0)
                                dur = interval.end - interval.begin
                                mid_point = interval.begin + (dur / 2)
                                if begin <= mid_point <= end:
                                    if interval.label not in phone_mapping:
                                        max_id += 1
                                        phone_mapping[interval.label] = max_id
                                        new_phones.append(
                                            {
                                                "id": max_id,
                                                "mapping_id": max_id - 1,
                                                "phone": interval.label,
                                                "kaldi_label": interval.label,
                                                "phone_type": PhoneType.extra,
                                            }
                                        )
                                    reference_intervals.append(
                                        {
                                            "id": interval_id,
                                            "begin": interval.begin,
                                            "end": interval.end,
                                            "phone_id": phone_mapping[interval.label],
                                            "utterance_id": u_id,
                                            "workflow_id": workflow.id,
                                        }
                                    )
                                    interval_id += 1
                                if mid_point > end:
                                    intervals[speaker_name].insert(0, interval)
                                    break
            if new_phones:
                session.execute(sqlalchemy.insert(Phone.__table__), new_phones)
                session.commit()
            session.execute(sqlalchemy.insert(PhoneInterval.__table__), reference_intervals)
            session.query(CorpusWorkflow).filter(CorpusWorkflow.id == workflow.id).update(
                {CorpusWorkflow.done: True, CorpusWorkflow.alignments_collected: True}
            )
            session.commit()

    def validate_corpus(self):
        """
        Validate the loaded files
        """
        if self.transcriptions_required:
            with self.session() as session:
                has_transcriptions = (
                    session.query(Utterance)
                    .filter(Utterance.text != None, Utterance.text != "")  # noqa
                    .first()
                    is not None
                )
            if not has_transcriptions:
                raise CorpusError(
                    "MFA could not find transcription files for the sound files, "
                    "please see "
                    "https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/corpus_structure.html "
                    "for details on how to structure your corpus"
                )

    def load_corpus(self) -> None:
        """
        Load the corpus
        """
        self.initialize_database()
        self._load_corpus()
        self.validate_corpus()
        self._create_dummy_dictionary()
        self.initialize_jobs()
        self.normalize_text()
        self.generate_features()

    def reset_features(self):
        with self.session() as session:
            logger.debug("Dropping indexes...")
            session.execute(sqlalchemy.text("DROP INDEX IF EXISTS utterance_xvector_index;"))
            session.execute(sqlalchemy.text("DROP INDEX IF EXISTS speaker_xvector_index;"))
            session.execute(sqlalchemy.text("DROP INDEX IF EXISTS utterance_ivector_index;"))
            session.execute(sqlalchemy.text("DROP INDEX IF EXISTS speaker_ivector_index;"))
            session.execute(sqlalchemy.text("DROP INDEX IF EXISTS utterance_plda_vector_index;"))
            session.execute(sqlalchemy.text("DROP INDEX IF EXISTS speaker_plda_vector_index;"))
            session.commit()
            logger.debug("Resetting utterance features...")
            session.execute(
                sqlalchemy.update(Corpus).values(
                    ivectors_calculated=False,
                    plda_calculated=False,
                    xvectors_loaded=False,
                    features_generated=False,
                )
            )
            session.execute(
                sqlalchemy.update(Utterance).values(
                    ivector=None, features=None, xvector=None, plda_vector=None
                )
            )
            session.execute(
                sqlalchemy.update(Speaker).values(
                    cmvn=None, fmllr=None, ivector=None, xvector=None, plda_vector=None
                )
            )
            session.commit()
        logger.debug("Deleting local files...")
        paths = [
            self.output_directory.joinpath("cmvn.ark"),
            self.output_directory.joinpath("cmvn.scp"),
            self.output_directory.joinpath("feats.scp"),
            self.output_directory.joinpath("ivectors.scp"),
        ]
        for path in paths:
            path.unlink(missing_ok=True)
        for j in self.jobs:
            paths = [
                j.construct_path(self.split_directory, "cmvn", "scp"),
                j.construct_path(self.split_directory, "ivectors", "scp"),
                j.construct_path(self.split_directory, "ivectors", "ark"),
            ]
            for path in paths:
                path.unlink(missing_ok=True)
            for d_id in j.dictionary_ids:
                paths = [
                    j.construct_path(self.split_directory, "trans", "scp", d_id),
                    j.construct_path(self.split_directory, "trans", "ark", d_id),
                    j.construct_path(self.split_directory, "cmvn", "scp", d_id),
                    j.construct_path(self.split_directory, "feats", "scp", d_id),
                    j.construct_path(self.split_directory, "feats", "ark", d_id),
                    j.construct_path(self.split_directory, "final_features", "scp", d_id),
                    j.construct_path(self.split_directory, "final_features", "ark", d_id),
                ]
                for path in paths:
                    path.unlink(missing_ok=True)

    def generate_final_features(self) -> None:
        """
        Generate features for the corpus
        """
        logger.info("Generating final features...")
        time_begin = time.time()
        log_directory = self.split_directory.joinpath("log")
        os.makedirs(log_directory, exist_ok=True)
        arguments = self.final_feature_arguments()
        for _ in run_kaldi_function(
            FinalFeatureFunction, arguments, total_count=self.num_utterances
        ):
            pass
        with self.session() as session:
            update_mapping = {}
            session.query(Utterance).update({"ignored": True})
            session.commit()
            for j in self.jobs:
                with mfa_open(j.feats_scp_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line == "":
                            continue
                        f = line.split(maxsplit=1)
                        utt_id = int(f[0].split("-")[-1])
                        feats = f[1]
                        update_mapping[utt_id] = {
                            "id": utt_id,
                            "features": feats,
                            "ignored": False,
                        }

            bulk_update(session, Utterance, list(update_mapping.values()))
            session.commit()

            non_ignored_check = (
                session.query(Utterance).filter(Utterance.ignored == False).first()  # noqa
            )
            if non_ignored_check is None:
                raise FeatureGenerationError(
                    f"No utterances had features, please check the logs in {log_directory} for errors."
                )
            ignored_utterances = (
                session.query(
                    SoundFile.sound_file_path,
                    Speaker.name,
                    Utterance.begin,
                    Utterance.end,
                    Utterance.text,
                )
                .join(Utterance.speaker)
                .join(Utterance.file)
                .join(File.sound_file)
                .filter(Utterance.ignored == True)  # noqa
            )
            ignored_count = 0
            for sound_file_path, speaker_name, begin, end, text in ignored_utterances:
                logger.debug(f"  - Ignored File: {sound_file_path}")
                logger.debug(f"    - Speaker: {speaker_name}")
                logger.debug(f"    - Begin: {begin}")
                logger.debug(f"    - End: {end}")
                logger.debug(f"    - Text: {text}")
                ignored_count += 1
            if ignored_count:
                logger.warning(
                    f"There were {ignored_count} utterances ignored due to an issue in feature generation, see the log file for full "
                    "details or run `mfa validate` on the corpus."
                )
        logger.debug(f"Generating final features took {time.time() - time_begin:.3f} seconds")

    def generate_features(self) -> None:
        """
        Generate features for the corpus
        """
        with self.session() as session:
            final_features_check = session.query(Corpus).first().features_generated
            if final_features_check:
                self.features_generated = True
                logger.info("Features already generated.")
                return
            feature_check = (
                session.query(Utterance).filter(Utterance.features != None).first()  # noqa
                is not None
            )
        if self.feature_type == "mfcc" and not feature_check:
            self.mfcc()
        self.combine_feats()
        if self.uses_cmvn:
            logger.info("Calculating CMVN...")
            self.calc_cmvn()
        if self.uses_voiced:
            self.compute_vad()
        self.generate_final_features()
        self._write_feats()
        self.features_generated = True
        with self.session() as session:
            session.query(Corpus).update({"features_generated": True})
            session.commit()
        self.create_corpus_split()

    def create_corpus_split(self) -> None:
        """Create the split directory for the corpus"""
        with self.session() as session:
            c = session.query(Corpus).first()
            c.current_subset = 0
            session.commit()
        logger.info("Creating corpus split...")
        super().create_corpus_split()

    def compute_vad_arguments(self) -> List[VadArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.corpus.features.ComputeVadFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.corpus.features.VadArguments`]
            Arguments for processing
        """
        return [
            VadArguments(
                j.id,
                getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                self.split_directory.joinpath("log", f"compute_vad.{j.id}.log"),
                self.vad_options,
            )
            for j in self.jobs
        ]

    def calc_fmllr_arguments(self, iteration: Optional[int] = None) -> List[CalcFmllrArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.corpus.features.CalcFmllrFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.corpus.features.CalcFmllrArguments`]
            Arguments for processing
        """
        base_log = "calc_fmllr"
        if iteration is not None:
            base_log += f".{iteration}"
        arguments = []
        for j in self.jobs:
            arguments.append(
                CalcFmllrArguments(
                    j.id,
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    self.working_log_directory.joinpath(f"{base_log}.{j.id}.log"),
                    self.working_directory,
                    self.alignment_model_path,
                    self.model_path,
                    self.fmllr_options,
                )
            )
        return arguments

    def mfcc_arguments(self) -> List[MfccArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.corpus.features.MfccFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.corpus.features.MfccArguments`]
            Arguments for processing
        """
        return [
            MfccArguments(
                j.id,
                getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                self.split_directory.joinpath("log", f"make_mfcc.{j.id}.log"),
                self.split_directory,
                self.mfcc_computer,
                self.pitch_computer,
            )
            for j in self.jobs
        ]

    def final_feature_arguments(self) -> List[FinalFeatureArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.corpus.features.MfccFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.corpus.features.MfccArguments`]
            Arguments for processing
        """
        return [
            FinalFeatureArguments(
                j.id,
                getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                self.split_directory.joinpath("log", f"generate_final_features.{j.id}.log"),
                self.split_directory,
                self.uses_cmvn,
                getattr(self, "sliding_cmvn", False),
                self.uses_voiced,
                getattr(self, "subsample", None),
            )
            for j in self.jobs
        ]

    def mfcc(self) -> None:
        """
        Multiprocessing function that converts sound files into MFCCs.

        See :kaldi_docs:`feat` for an overview on feature generation in Kaldi.

        See Also
        --------
        :class:`~montreal_forced_aligner.corpus.features.MfccFunction`
            Multiprocessing helper function for each job
        :meth:`.AcousticCorpusMixin.mfcc_arguments`
            Job method for generating arguments for helper function
        :kaldi_steps:`make_mfcc`
            Reference Kaldi script
        """
        logger.info("Generating MFCCs...")
        begin = time.time()
        log_directory = self.split_directory.joinpath("log")
        os.makedirs(log_directory, exist_ok=True)
        arguments = self.mfcc_arguments()
        for _ in run_kaldi_function(MfccFunction, arguments, total_count=self.num_utterances):
            pass
        logger.debug(f"Generating MFCCs took {time.time() - begin:.3f} seconds")

    def calc_cmvn(self) -> None:
        """
        Calculate CMVN statistics for speakers

        See Also
        --------
        :kaldi_src:`compute-cmvn-stats`
            Relevant Kaldi binary
        """
        self._write_spk2utt()
        spk2utt_path = self.corpus_output_directory.joinpath("spk2utt.scp")
        feats_scp_path = self.corpus_output_directory.joinpath("feats.scp")
        cmvn_ark_path = self.corpus_output_directory.joinpath("cmvn.ark")
        log_path = self.features_log_directory.joinpath("cmvn.log")
        with kalpy_logger("kalpy.cmvn", log_path) as cmvn_logger:
            cmvn_logger.info(f"Reading features from: {feats_scp_path}")
            cmvn_logger.info(f"Reading spk2utt from: {spk2utt_path}")
            spk2utt = KaldiMapping(list_mapping=True)
            spk2utt.load(spk2utt_path)
            mfcc_archive = FeatureArchive(feats_scp_path)
            computer = CmvnComputer()
            computer.export_cmvn(cmvn_ark_path, mfcc_archive, spk2utt, write_scp=True)
            mfcc_archive.close()
        update_mapping = []
        cmvn_scp = self.corpus_output_directory.joinpath("cmvn.scp")
        with self.session() as session:
            for s, cmvn in load_scp(cmvn_scp).items():
                if isinstance(cmvn, list):
                    cmvn = " ".join(cmvn)
                update_mapping.append({"id": int(s), "cmvn": cmvn})
            bulk_update(session, Speaker, update_mapping)
            session.commit()
            for j in self.jobs:
                query = (
                    session.query(Speaker.id, Speaker.cmvn)
                    .join(Speaker.utterances)
                    .filter(Speaker.cmvn != None, Utterance.job_id == j.id)  # noqa
                    .distinct()
                )
                with mfa_open(j.construct_path(self.split_directory, "cmvn", "scp"), "w") as f:
                    for s_id, cmvn in sorted(query, key=lambda x: str(x)):
                        f.write(f"{s_id} {cmvn}\n")

    def calc_fmllr(self, iteration: Optional[int] = None) -> None:
        """
        Multiprocessing function that computes speaker adaptation transforms via
        feature-space Maximum Likelihood Linear Regression (fMLLR).

        See Also
        --------
        :class:`~montreal_forced_aligner.corpus.features.CalcFmllrFunction`
            Multiprocessing helper function for each job
        :meth:`.AcousticCorpusMixin.calc_fmllr_arguments`
            Job method for generating arguments for the helper function
        :kaldi_steps:`align_fmllr`
            Reference Kaldi script
        :kaldi_steps:`train_sat`
            Reference Kaldi script
        """
        begin = time.time()
        logger.info("Calculating fMLLR for speaker adaptation...")

        with self.session() as session:
            corpus = session.query(Corpus).first()
            num_utterances = corpus.current_subset
        if not num_utterances:
            num_utterances = self.num_utterances

        arguments = self.calc_fmllr_arguments(iteration=iteration)
        for _ in run_kaldi_function(CalcFmllrFunction, arguments, total_count=num_utterances):
            pass

        self.uses_speaker_adaptation = True
        update_mapping = []
        if not config.SINGLE_SPEAKER:
            for j in self.jobs:
                for d_id in j.dictionary_ids:
                    scp_p = j.construct_path(self.split_directory, "trans", "scp", d_id)
                    if not scp_p.exists():
                        continue
                    with mfa_open(scp_p) as f:
                        for line in f:
                            line = line.strip()
                            speaker, ark = line.split(maxsplit=1)
                            speaker = int(speaker)
                            update_mapping.append({"id": speaker, "fmllr": ark})
            with self.session() as session:
                bulk_update(session, Speaker, update_mapping)
                session.commit()
        logger.debug(f"Fmllr calculation took {time.time() - begin:.3f} seconds")

    def compute_vad(self) -> None:
        """
        Compute Voice Activity Detection features over the corpus

        See Also
        --------
        :class:`~montreal_forced_aligner.corpus.features.ComputeVadFunction`
            Multiprocessing helper function for each job
        :meth:`.AcousticCorpusMixin.compute_vad_arguments`
            Job method for generating arguments for helper function
        """
        with self.session() as session:
            c = session.query(Corpus).first()
            if c.vad_calculated:
                logger.info("VAD already computed, skipping!")
                return
        begin = time.time()
        logger.info("Computing VAD...")

        arguments = self.compute_vad_arguments()
        for _ in run_kaldi_function(
            ComputeVadFunction, arguments, total_count=self.num_utterances
        ):
            pass
        vad_lines = []
        utterance_mapping = []
        for j in self.jobs:
            vad_scp_path = j.construct_path(self.split_directory, "vad", "scp")
            with mfa_open(vad_scp_path) as inf:
                for line in inf:
                    vad_lines.append(line)
                    utt_id, ark = line.strip().split(maxsplit=1)
                    utt_id = int(utt_id.split("-")[-1])
                    utterance_mapping.append({"id": utt_id, "vad_ark": ark})
        with self.session() as session:
            bulk_update(session, Utterance, utterance_mapping)
            session.query(Corpus).update({Corpus.vad_calculated: True})
            session.commit()
        with mfa_open(self.corpus_output_directory.joinpath("vad.scp"), "w") as outf:
            for line in sorted(vad_lines, key=lambda x: x.split(maxsplit=1)[0]):
                outf.write(line)
        logger.debug(f"VAD computation took {time.time() - begin:.3f} seconds")

    def combine_feats(self) -> None:
        """
        Combine feature generation results and store relevant information
        """
        lines = []
        for j in self.jobs:
            with mfa_open(j.feats_scp_path) as f:
                for line in f:
                    lines.append(line)
        with open(self.corpus_output_directory.joinpath("feats.scp"), "w", encoding="utf8") as f:
            for line in sorted(lines, key=lambda x: x.split(maxsplit=1)[0]):
                f.write(line)

    def _write_feats(self) -> None:
        """Write feats scp file for Kaldi"""
        with self.session() as session, open(
            self.corpus_output_directory.joinpath("feats.scp"), "w", encoding="utf8"
        ) as f:
            utterances = (
                session.query(Utterance.kaldi_id, Utterance.features)
                .filter_by(ignored=False)
                .order_by(Utterance.kaldi_id)
            )
            for u_id, features in utterances:
                f.write(f"{u_id} {features}\n")

    def get_feat_dim(self) -> int:
        """
        Calculate the feature dimension for the corpus

        Returns
        -------
        int
            Dimension of feature vectors
        """
        job = self.jobs[0]
        dict_id = None
        if job.dictionary_ids:
            dict_id = self.jobs[0].dictionary_ids[0]
        feature_archive = job.construct_feature_archive(self.working_directory, dict_id)
        feat_dim = None
        for _, feats in feature_archive:
            feat_dim = feats.NumCols()
            break
        return feat_dim

    def _load_corpus_from_source_mp(self) -> None:
        """
        Load a corpus using multiprocessing
        """
        begin_time = time.process_time()
        job_queue = Queue()
        return_queue = Queue()
        finished_adding = threading.Event()
        stopped = threading.Event()
        file_counts = Counter()
        error_dict = {}
        procs = []
        parser = AcousticDirectoryParser(
            self.corpus_directory,
            job_queue,
            self.audio_directory,
            stopped,
            finished_adding,
            file_counts,
        )
        parser.start()
        for i in range(config.NUM_JOBS):
            p = CorpusProcessWorker(
                i,
                job_queue,
                return_queue,
                stopped,
                finished_adding,
                self.speaker_characters,
                self.sample_frequency,
            )
            procs.append(p)
            p.start()
        last_poll = time.time() - 30
        try:
            with self.session() as session, tqdm(total=100, disable=config.QUIET) as pbar:
                import_data = DatabaseImportData()
                while True:
                    try:
                        file = return_queue.get(timeout=1)
                        if isinstance(file, tuple):
                            error_type = file[0]
                            error = file[1]
                            if error_type == "error":
                                error_dict[error_type] = error
                            else:
                                if error_type not in error_dict:
                                    error_dict[error_type] = []
                                error_dict[error_type].append(error)
                            continue
                        if self.stopped.is_set():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished_processing.is_set():
                                break
                        else:
                            break
                        continue
                    if time.time() - last_poll > 5:
                        pbar.total = file_counts.value()
                        last_poll = time.time()
                    pbar.update(1)
                    import_data.add_objects(self.generate_import_objects(file))
                    return_queue.task_done()

                logger.debug(f"Processing queue: {time.process_time() - begin_time}")

                if "error" in error_dict:
                    session.rollback()
                    raise error_dict["error"]
                self._finalize_load(session, import_data)
            for k in ["sound_file_errors", "decode_error_files", "textgrid_read_errors"]:
                if hasattr(self, k):
                    if k in error_dict:
                        logger.info(
                            "There were some issues with files in the corpus. "
                            "Please look at the log file or run the validator for more information."
                        )
                        logger.debug(f"{k} showed {len(error_dict[k])} errors:")
                        if k in {"textgrid_read_errors", "sound_file_errors"}:
                            getattr(self, k).extend(error_dict[k])
                            for e in error_dict[k]:
                                logger.debug(f"{e.file_name}: {e.error}")
                        else:
                            logger.debug(", ".join(str(error_dict[k])))
                            setattr(self, k, error_dict[k])

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                logger.info(
                    "Detected ctrl-c, please wait a moment while we clean everything up..."
                )
            self.stopped.set()
            finished_adding.set()
            while True:
                try:
                    _ = job_queue.get(timeout=1)
                    if self.stopped.is_set():
                        continue
                    job_queue.task_done()
                except Empty:
                    for proc in procs:
                        if not proc.finished_processing.is_set():
                            break
                    else:
                        break
                try:
                    _ = return_queue.get(timeout=1)
                    _ = job_queue.get(timeout=1)
                    if self.stopped.is_set():
                        continue
                    return_queue.task_done()
                    job_queue.task_done()
                except Empty:
                    for proc in procs:
                        if not proc.finished_processing.is_set():
                            break
                    else:
                        break
            raise
        finally:
            parser.join()
            for p in procs:
                p.join()
            if self.stopped.is_set():
                logger.info(f"Stopped parsing early ({time.process_time() - begin_time} seconds)")
            else:
                logger.debug(
                    f"Parsed corpus directory with {config.NUM_JOBS} jobs in {time.process_time() - begin_time} seconds"
                )

    def _load_corpus_from_source(self) -> None:
        """
        Load a corpus without using multiprocessing
        """
        begin_time = time.time()

        all_sound_files = {}
        use_audio_directory = False
        if self.audio_directory and os.path.exists(self.audio_directory):
            use_audio_directory = True
            for root, _, files in os.walk(self.audio_directory, followlinks=True):
                if self.stopped.is_set():
                    return
                if root.startswith("."):  # Ignore hidden directories
                    continue
                exts = find_exts(files)
                exts.wav_files = {k: os.path.join(root, v) for k, v in exts.wav_files.items()}
                exts.other_audio_files = {
                    k: os.path.join(root, v) for k, v in exts.other_audio_files.items()
                }
                all_sound_files.update(exts.other_audio_files)
                all_sound_files.update(exts.wav_files)
        logger.debug(f"Walking through {self.corpus_directory}...")
        with self.session() as session:
            import_data = DatabaseImportData()
            for root, _, files in os.walk(self.corpus_directory, followlinks=True):
                if self.stopped.is_set():
                    return
                if root.startswith("."):  # Ignore hidden directories
                    continue
                exts = find_exts(files)
                relative_path = (
                    root.replace(str(self.corpus_directory), "").lstrip("/").lstrip("\\")
                )
                if not use_audio_directory:
                    all_sound_files = {}
                    wav_files = {k: os.path.join(root, v) for k, v in exts.wav_files.items()}
                    other_audio_files = {
                        k: os.path.join(root, v) for k, v in exts.other_audio_files.items()
                    }
                    all_sound_files.update(other_audio_files)
                    all_sound_files.update(wav_files)
                for file_name in exts.identifiers:
                    wav_path = None
                    transcription_path = None
                    if file_name in all_sound_files:
                        wav_path = all_sound_files[file_name]
                    if file_name in exts.lab_files:
                        lab_name = exts.lab_files[file_name]
                        transcription_path = os.path.join(root, lab_name)
                    elif file_name in exts.textgrid_files:
                        tg_name = exts.textgrid_files[file_name]
                        transcription_path = os.path.join(root, tg_name)
                    if wav_path is None:  # Not a file for MFA
                        continue
                    try:
                        file = FileData.parse_file(
                            file_name,
                            wav_path,
                            transcription_path,
                            relative_path,
                            self.speaker_characters,
                            self.sample_frequency,
                        )
                        import_data.add_objects(self.generate_import_objects(file))
                    except TextParseError as e:
                        self.decode_error_files.append(e)
                    except TextGridParseError as e:
                        self.textgrid_read_errors.append(e)
                    except SoundFileError as e:
                        self.sound_file_errors.append(e)
            self._finalize_load(session, import_data)
        if self.decode_error_files or self.textgrid_read_errors:
            logger.info(
                "There were some issues with files in the corpus. "
                "Please look at the log file or run the validator for more information."
            )
            if self.decode_error_files:
                logger.debug(
                    f"There were {len(self.decode_error_files)} errors decoding text files:"
                )
                logger.debug(", ".join(self.decode_error_files))
            if self.textgrid_read_errors:
                logger.debug(
                    f"There were {len(self.textgrid_read_errors)} errors decoding reading TextGrid files:"
                )
                for e in self.textgrid_read_errors:
                    logger.debug(f"{e.file_name}: {e.error}")

        logger.debug(f"Parsed corpus directory in {time.time() - begin_time:.3f} seconds")


class AcousticCorpusPronunciationMixin(
    AcousticCorpusMixin, MultispeakerDictionaryMixin, metaclass=ABCMeta
):
    """
    Mixin for acoustic corpora with Pronunciation dictionaries

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusMixin`
        For corpus parsing parameters
    :class:`~montreal_forced_aligner.dictionary.multispeaker.MultispeakerDictionaryMixin`
        For dictionary parsing parameters
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_corpus(self) -> None:
        """
        Load the corpus
        """
        all_begin = time.time()
        self.initialize_database()
        if self.dictionary_model is not None and not self.imported:
            self.dictionary_setup()
            logger.debug(f"Loaded dictionary in {time.time() - all_begin:.3f} seconds")

        begin = time.time()
        self._load_corpus()
        logger.debug(f"Loaded corpus in {time.time() - begin:.3f} seconds")

        begin = time.time()
        self.initialize_jobs()
        logger.debug(f"Initialized jobs in {time.time() - begin:.3f} seconds")

        initialized_check = self.text_normalized

        self.normalize_text()

        if self.dictionary_model is not None and not initialized_check:
            self.apply_phonological_rules()
            self.calculate_disambiguation()
            self.calculate_phone_mapping()

            begin = time.time()
            self.write_lexicon_information()
            logger.debug(f"Wrote lexicon information in {time.time() - begin:.3f} seconds")
        else:
            self.load_phone_topologies()
            self.load_phone_groups()
            self.load_lexicon_compilers()

        begin = time.time()
        self.generate_features()
        logger.debug(f"Generated features in {time.time() - begin:.3f} seconds")

        logger.debug(f"Setting up corpus took {time.time() - all_begin:.3f} seconds")

    def subset_lexicon(self, write_disambiguation: Optional[bool] = False) -> None:
        included_words = set()
        with self.session() as session:
            corpus = session.query(Corpus).first()
            session.execute(
                sqlalchemy.update(Word)
                .where(Word.word_type == WordType.speech)
                .values(included=False)
            )
            session.flush()
            if corpus.current_subset > 0:
                subset_utterances = (
                    session.query(Utterance.normalized_text)
                    .filter(Utterance.in_subset == True)  # noqa
                    .filter(Utterance.ignored == False)  # noqa
                )
                for (u_text,) in subset_utterances:
                    included_words.update(u_text.split())
                session.execute(
                    sqlalchemy.update(Word)
                    .where(Word.word_type == WordType.speech)
                    .where(Word.count > self.oov_count_threshold)
                    .where(Word.word.in_(included_words))
                    .values(included=True)
                )
            else:
                session.execute(
                    sqlalchemy.update(Word)
                    .where(Word.word_type == WordType.speech)
                    .where(Word.count > self.oov_count_threshold)
                    .values(included=True)
                )
            session.commit()
        self.write_lexicon_information(write_disambiguation=write_disambiguation)


class AcousticCorpus(AcousticCorpusMixin, DictionaryMixin, MfaWorker):
    """
    Standalone class for working with acoustic corpora and pronunciation dictionaries

    Most functionality in MFA will use the :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusPronunciationMixin` class instead of this class.

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusPronunciationMixin`
        For dictionary and corpus parsing parameters
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For MFA processing parameters
    :class:`~montreal_forced_aligner.abc.TemporaryDirectoryMixin`
        For temporary directory parameters
    """

    def __init__(self, **kwargs):
        super(AcousticCorpus, self).__init__(**kwargs)

    @property
    def identifier(self) -> str:
        """Identifier for the corpus"""
        return self.data_source_identifier

    @property
    def output_directory(self) -> Path:
        """Root temporary directory to store corpus and dictionary files"""
        return config.TEMPORARY_DIRECTORY.joinpath(self.identifier)

    @property
    def working_directory(self) -> Path:
        """Working directory to save temporary corpus and dictionary files"""
        return self.corpus_output_directory


class AcousticCorpusWithPronunciations(AcousticCorpusPronunciationMixin, MfaWorker):
    """
    Standalone class for parsing an acoustic corpus with a pronunciation dictionary
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def identifier(self) -> str:
        """Identifier for the corpus"""
        return self.data_source_identifier

    @property
    def output_directory(self) -> Path:
        """Root temporary directory to store corpus and dictionary files"""
        return config.TEMPORARY_DIRECTORY.joinpath(self.identifier)

    @property
    def working_directory(self) -> Path:
        """Working directory to save temporary corpus and dictionary files"""
        return self.output_directory
