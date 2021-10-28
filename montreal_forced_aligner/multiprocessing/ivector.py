from __future__ import annotations
from typing import TYPE_CHECKING, Union, Any, List, Dict, Optional
if TYPE_CHECKING:
    from ..trainers.ivector_extractor import IvectorExtractorTrainer, IvectorConfigType
    from ..speaker_classifier import SpeakerClassifier
    from ..segmenter import SegmentationType
    from ..segmenter import Segmenter
    from ..config.speaker_classification_config import SpeakerClassificationConfig
    from ..config import ConfigDict
    from .alignment import IterationType
    from ..corpus import CorpusType
    from ..corpus.classes import Utterance, File, Speaker
import subprocess
import multiprocessing as mp
import os
from ..utils import thirdparty_binary
from .helper import run_mp, run_non_mp
from ..helper import load_scp


def gmm_gselect_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ivector_options: ConfigDict,
    dubm_path: str,
    gselect_paths: Dict[str, str]) -> None:
    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            gselect_path = gselect_paths[dict_name]
            subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                     f"--n={ivector_options['subsample']}",
                                                     feature_string,
                                                     'ark:-'],
                                                    stdout=subprocess.PIPE,
                                                    stderr=log_file, env=os.environ)

            gselect_proc = subprocess.Popen([thirdparty_binary('gmm-gselect'),
                                             f"--n={ivector_options['num_gselect']}",
                                             dubm_path,
                                             'ark:-',
                                             f'ark:{gselect_path}'],
                                            stdin=subsample_feats_proc.stdout,
                                            stderr=log_file, env=os.environ)
            gselect_proc.communicate()


def gmm_gselect(aligner: IvectorExtractorTrainer) -> None:
    """
    Multiprocessing function that stores Gaussian selection indices on disk

    See:

    - http://kaldi-asr.org/doc/gmm-gselect_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_diag_ubm.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    config : :class:`~aligner.config.DiagUbmConfig`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation

    """
    jobs = [x.gmm_gselect_arguments(aligner) for x in aligner.corpus.jobs]
    if aligner.use_mp:
        run_mp(gmm_gselect_func, jobs, aligner.working_log_directory)
    else:
        run_non_mp(gmm_gselect_func, jobs, aligner.working_log_directory)


def acc_global_stats_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ivector_options: ConfigDict,
    gselect_paths: Dict[str, str],
    acc_paths: Dict[str, str],
    dubm_path: str) -> None:
    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            gselect_path = gselect_paths[dict_name]
            acc_path = acc_paths[dict_name]
            subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                     f"--n={ivector_options['subsample']}",
                                                     feature_string,
                                                     'ark:-'],
                                                    stdout=subprocess.PIPE,
                                                    stderr=log_file, env=os.environ)
            gmm_global_acc_proc = subprocess.Popen([thirdparty_binary('gmm-global-acc-stats'),
                                                    f'--gselect=ark:{gselect_path}',
                                                    dubm_path,
                                                    'ark:-',
                                                    acc_path],
                                                   stderr=log_file,
                                                   stdin=subsample_feats_proc.stdout, env=os.environ)
            gmm_global_acc_proc.communicate()


def acc_global_stats(aligner: IvectorExtractorTrainer) -> None:
    """
    Multiprocessing function that accumulates global GMM stats

    See:

    - http://kaldi-asr.org/doc/gmm-global-acc-stats_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_diag_ubm.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    config : :class:`~aligner.config.DiagUbmConfig`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation
    iteration : int
        Iteration to calculate stats for
    """
    jobs = [x.acc_global_stats_arguments(aligner) for x in aligner.corpus.jobs]
    if aligner.use_mp:
        run_mp(acc_global_stats_func, jobs, aligner.working_log_directory)
    else:
        run_non_mp(acc_global_stats_func, jobs, aligner.working_log_directory)

    # Don't remove low-count Gaussians till the last tier,
    # or gselect info won't be valid anymore
    if aligner.iteration < aligner.ubm_num_iterations:
        opt = '--remove-low-count-gaussians=false'
    else:
        opt = f'--remove-low-count-gaussians={aligner.ubm_remove_low_count_gaussians}'
    log_path = os.path.join(aligner.working_log_directory, f'update.{aligner.iteration}.log')
    with open(log_path, 'w') as log_file:
        acc_files = []
        for j in jobs:
            acc_files.extend(j.acc_paths.values())
        sum_proc = subprocess.Popen([thirdparty_binary('gmm-global-sum-accs'),
                                     '-'] + acc_files,
                                    stderr=log_file, stdout=subprocess.PIPE, env=os.environ)
        gmm_global_est_proc = subprocess.Popen([thirdparty_binary('gmm-global-est'),
                                                opt,
                                                f'--min-gaussian-weight={aligner.ubm_min_gaussian_weight}',
                                                aligner.current_dubm_path,
                                                "-",
                                                aligner.next_dubm_path],
                                               stderr=log_file, stdin=sum_proc.stdout, env=os.environ)
        gmm_global_est_proc.communicate()
        # Clean up
        if not aligner.debug:
            for p in acc_files:
                os.remove(p)


def gauss_to_post_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ivector_options: ConfigDict,
    post_paths: Dict[str, str],
    dubm_path: str):
    modified_posterior_scale = ivector_options['posterior_scale'] * ivector_options['subsample']
    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            post_path = post_paths[dict_name]
            subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                     f"--n={ivector_options['subsample']}",
                                                     feature_string,
                                                     'ark:-'],
                                                    stdout=subprocess.PIPE,
                                                    stderr=log_file, env=os.environ)
            gmm_global_get_post_proc = subprocess.Popen([thirdparty_binary('gmm-global-get-post'),
                                                         f"--n={ivector_options['num_gselect']}",
                                                         f"--min-post={ivector_options['min_post']}",
                                                         dubm_path,
                                                         'ark:-',
                                                         'ark:-'],
                                                        stdout=subprocess.PIPE,
                                                        stdin=subsample_feats_proc.stdout,
                                                        stderr=log_file, env=os.environ)
            scale_post_proc = subprocess.Popen([thirdparty_binary('scale-post'),
                                                'ark:-',
                                                str(modified_posterior_scale),
                                                f'ark:{post_path}'],
                                               stdin=gmm_global_get_post_proc.stdout,
                                               stderr=log_file, env=os.environ)
            scale_post_proc.communicate()


def gauss_to_post(aligner: IvectorExtractorTrainer) -> None:
    """
    Multiprocessing function that does Gaussian selection and posterior extraction

    See:

    - http://kaldi-asr.org/doc/gmm-global-get-post_8cc.html
    - http://kaldi-asr.org/doc/scale-post_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/online/nnet2/train_ivector_extractor.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    config : :class:`~aligner.config.iVectorExtractorConfig`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation
    """
    jobs = [x.gauss_to_post_arguments(aligner) for x in aligner.corpus.jobs]
    if aligner.use_mp:
        run_mp(gauss_to_post_func, jobs, aligner.working_log_directory)
    else:
        run_non_mp(gauss_to_post_func, jobs, aligner.working_log_directory)


def acc_ivector_stats_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ivector_options: ConfigDict,
    ie_path: str,
    post_paths: Dict[str, str],
    acc_init_paths: Dict[str, str]) -> None:
    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            post_path = post_paths[dict_name]
            acc_init_path = acc_init_paths[dict_name]
            subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                     f"--n={ivector_options['subsample']}",
                                                     feature_string,
                                                     'ark:-'],
                                                    stdout=subprocess.PIPE,
                                                    stderr=log_file, env=os.environ)
            acc_stats_proc = subprocess.Popen([thirdparty_binary('ivector-extractor-acc-stats'),
                                               '--num-threads=1',
                                               ie_path,
                                               'ark:-',
                                               f'ark:{post_path}',
                                               acc_init_path],
                                              stdin=subsample_feats_proc.stdout,
                                              stderr=log_file, env=os.environ)
            acc_stats_proc.communicate()


def acc_ivector_stats(trainer: IvectorExtractorTrainer) -> None:
    """
    Multiprocessing function that calculates job_name-vector extractor stats

    See:

    - http://kaldi-asr.org/doc/ivector-extractor-acc-stats_8cc.html
    - http://kaldi-asr.org/doc/ivector-extractor-sum-accs_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/online/nnet2/train_ivector_extractor.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    config : :class:`~aligner.config.iVectorExtractorConfig`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation
    iteration : int
        Iteration to calculate stats for
    """

    jobs = [x.ivector_acc_stats_arguments(trainer) for x in trainer.corpus.jobs]
    if trainer.use_mp:
        run_mp(acc_ivector_stats_func, jobs, trainer.working_log_directory)
    else:
        run_non_mp(acc_ivector_stats_func, jobs, trainer.working_log_directory)

    log_path = os.path.join(trainer.working_log_directory, f'sum_acc.{trainer.iteration}.log')
    acc_path = os.path.join(trainer.working_directory, f'acc.{trainer.iteration}')
    with open(log_path, 'w', encoding='utf8') as log_file:
        accinits = []
        for j in jobs:
            accinits.extend(j.acc_init_paths.values())
        sum_accs_proc = subprocess.Popen([thirdparty_binary('ivector-extractor-sum-accs'),
                                          '--parallel=true']
                                         + accinits
                                         + [acc_path],
                                         stderr=log_file, env=os.environ)

        sum_accs_proc.communicate()
    # clean up
    for p in accinits:
        os.remove(p)
        # Est extractor
    log_path = os.path.join(trainer.working_log_directory, f'update.{trainer.iteration}.log')
    with open(log_path, 'w') as log_file:
        extractor_est_proc = subprocess.Popen([thirdparty_binary('ivector-extractor-est'),
                                               f'--num-threads={trainer.corpus.num_jobs}',
                                               f'--gaussian-min-count={trainer.gaussian_min_count}',
                                               trainer.current_ie_path,
                                               os.path.join(trainer.working_directory, f'acc.{trainer.iteration}'),
                                               trainer.next_ie_path],
                                              stderr=log_file, env=os.environ)
        extractor_est_proc.communicate()


def extract_ivectors_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ivector_options: ConfigDict,
    ali_paths: Dict[str, str],
    ie_path: str,
    ivector_paths: Dict[str, str],
    weight_paths: Dict[str, str],
    model_path: str,
    dubm_path: str) -> None:
    """
    Parameters
    ----------
    config : :class:`~aligner.trainers.IvectorExtractorTrainer`
        Configuration object for training
    job_name : int
        Job identifier
    """

    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            ali_path = ali_paths[dict_name]
            weight_path = weight_paths[dict_name]
            ivectors_path = ivector_paths[dict_name]
            feature_string = feature_strings[dict_name]
            use_align = os.path.exists(ali_path)
            if use_align:
                ali_to_post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                                     f'ark:{ali_path}', 'ark:-'],
                                                    stderr=log_file,
                                                    stdout=subprocess.PIPE, env=os.environ)
                weight_silence_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'),
                                                        str(ivector_options['silence_weight']),
                                                        ivector_options['sil_phones'],
                                                        model_path,
                                                        'ark:-', 'ark:-'],
                                                       stderr=log_file,
                                                       stdin=ali_to_post_proc.stdout,
                                                       stdout=subprocess.PIPE, env=os.environ)
                post_to_weight_proc = subprocess.Popen([thirdparty_binary('post-to-weights'),
                                                        'ark:-', f'ark:{weight_path}'],
                                                       stderr=log_file,
                                                       stdin=weight_silence_proc.stdout, env=os.environ)
                post_to_weight_proc.communicate()

            gmm_global_get_post_proc = subprocess.Popen([thirdparty_binary('gmm-global-get-post'),
                                                         f"--n={ivector_options['num_gselect']}",
                                                         f"--min-post={ivector_options['min_post']}",
                                                         dubm_path,
                                                         feature_string,
                                                         'ark:-'],
                                                        stdout=subprocess.PIPE,
                                                        stderr=log_file, env=os.environ)
            if use_align:
                weight_proc = subprocess.Popen([thirdparty_binary('weight-post'),
                                                'ark:-', f'ark,s,cs:{weight_path}', 'ark:-'],
                                               stdin=gmm_global_get_post_proc.stdout,
                                               stdout=subprocess.PIPE, stderr=log_file, env=os.environ)
                extract_in = weight_proc.stdout
            else:
                extract_in = gmm_global_get_post_proc.stdout
            extract_proc = subprocess.Popen([thirdparty_binary('ivector-extract'),
                                             f"--acoustic-weight={ivector_options['posterior_scale']}",
                                             '--compute-objf-change=true',
                                             f"--max-count={ivector_options['max_count']}",
                                             ie_path,
                                             feature_string,
                                             'ark,s,cs:-',
                                             f'ark,t:{ivectors_path}'],
                                            stderr=log_file,
                                            stdin=extract_in, env=os.environ)
            extract_proc.communicate()


def extract_ivectors(ivector_extractor: Union[SpeakerClassifier, IvectorExtractorTrainer]) -> None:
    """
    Multiprocessing function that extracts job_name-vectors.

    See:

    - http://kaldi-asr.org/doc/ivector-extract-online2_8cc.html
    - http://kaldi-asr.org/doc/copy-feats_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/online/nnet2/extract_ivectors_online.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    config : :class:`~montreal_forced_aligner.config.iVectorExtractorConfig`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation
    """

    log_dir = ivector_extractor.log_directory
    os.makedirs(log_dir, exist_ok=True)
    func = extract_ivectors_func
    jobs = [x.extract_ivector_arguments(ivector_extractor) for x in ivector_extractor.corpus.jobs]
    if ivector_extractor.use_mp:
        run_mp(func, jobs, log_dir)
    else:
        run_non_mp(func, jobs, log_dir)



def get_initial_segmentation(frames: List[Union[int, str]], frame_shift: int) -> SegmentationType:
    segs = []
    cur_seg = None
    silent_frames = 0
    non_silent_frames = 0
    for i, f in enumerate(frames):
        if int(f) > 0:
            non_silent_frames += 1
            if cur_seg is None:
                cur_seg = {'begin': i * frame_shift}
        else:
            silent_frames += 1
            if cur_seg is not None:
                cur_seg['end'] = (i - 1) * frame_shift
                segs.append(cur_seg)
                cur_seg = None
    if cur_seg is not None:
        cur_seg['end'] = len(frames) * frame_shift
        segs.append(cur_seg)
    return segs


def merge_segments(segments: SegmentationType, min_pause_duration: float, max_segment_length: float, snap_boundary_threshold: float) -> SegmentationType:
    merged_segs = []
    for s in segments:
        if not merged_segs or s['begin'] > merged_segs[-1]['end'] + min_pause_duration or \
                s['end'] - merged_segs[-1]['begin'] > max_segment_length:
            if s['end'] - s['begin'] > min_pause_duration:
                if merged_segs and snap_boundary_threshold:
                    boundary_gap = s['begin'] - merged_segs[-1]['end']
                    if boundary_gap < snap_boundary_threshold:
                        half_boundary = boundary_gap / 2
                    else:
                        half_boundary = snap_boundary_threshold / 2
                    merged_segs[-1]['end'] += half_boundary
                    s['begin'] -= half_boundary

                merged_segs.append(s)
        else:
            merged_segs[-1]['end'] = s['end']
    return merged_segs


def segment_vad_func(
    log_path: str,
    dictionaries: List[str],
    vad_paths: Dict[str, str],
    segmentation_options: ConfigDict) -> Dict[str, Utterance]:
    from ..corpus.classes import Utterance, File, Speaker
    utterances = {}
    speaker = Speaker('speech')
    for dict_name in dictionaries:
        vad_path = vad_paths[dict_name]

        vad = load_scp(vad_path, data_type=int)
        for recording, frames in vad.items():
            file = File(recording)
            initial_segments = get_initial_segmentation(frames, segmentation_options['frame_shift'])
            merged = merge_segments(initial_segments, segmentation_options['min_pause_duration'],
                                    segmentation_options['max_segment_length'], segmentation_options['snap_boundary_threshold'])
            for seg in merged:
                utterances[recording] = Utterance(speaker, file, begin=seg['begin'], end=seg['end'])
    return utterances

def segment_vad(segmenter: Segmenter) -> None:
    from ..corpus.classes import Speaker
    jobs = [x.segments_vad_arguments(segmenter) for x in segmenter.corpus.jobs]
    if segmenter.segmentation_config.use_mp:
        manager = mp.Manager()
        segment_info = manager.dict()
        run_mp(segment_vad_func, jobs, segmenter.corpus.features_log_directory, segment_info)
    else:
        segment_info = {}
        segment_info = run_non_mp(segment_vad_func, jobs, segmenter.corpus.features_log_directory, segment_info)

    for j in range(segmenter.corpus.num_jobs):
        for old_utt, utterance in segment_info[j].items():
            old_utt = segmenter.corpus.utterances[old_utt]
            file = old_utt.file
            if utterance.speaker_name not in segmenter.corpus.speakers:
                segmenter.corpus.speakers[utterance.speaker_name] = Speaker(utterance.speaker_name)
            speaker = segmenter.corpus.speakers[utterance.speaker_name]
            utterance.file = file
            utterance.set_speaker(speaker)
            segmenter.corpus.delete_utterance(old_utt)

def classify_speakers_func(log_path: str,
    dictionaries: List[str],
    model_path: str,
    labels_path: str,
    ivector_paths: Dict[str, str]) -> Dict[str, str]:
    from ..helper import load_scp
    from joblib import load
    import numpy as np
    import warnings
    speakers = {}
    with open(labels_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split()
            speaker, speak_ind = line
            speakers[int(speak_ind)] = speaker
    utt_speak_mapping = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = load(model_path)
    for dict_name in dictionaries:
        ivectors_path = ivector_paths[dict_name]
        ivec = load_scp(ivectors_path)
        x = []
        for utt, ivector in ivec.items():
            ivector = [float(x) for x in ivector]
            x.append(ivector)
        x = np.array(x)
        y = clf.predict(x)
        for i, utt in enumerate(ivec.keys()):
            speak_ind = y[i]
            speaker = speakers[speak_ind]
            utt_speak_mapping[utt] = speaker
    return utt_speak_mapping


def classify_speakers(speaker_classifier: SpeakerClassifier) -> None:
    from ..corpus.classes import Speaker
    log_directory = speaker_classifier.working_log_directory
    jobs = [x.classify_speaker_arguments(speaker_classifier) for x in speaker_classifier.corpus.jobs]

    if speaker_classifier.use_mp:
        manager = mp.Manager()
        speaker_info = manager.dict()
        run_mp(classify_speakers_func, jobs, log_directory, speaker_info)
    else:
        speaker_info = {}
        speaker_info = run_non_mp(classify_speakers_func, jobs, log_directory, speaker_info)
    for j in range(speaker_classifier.corpus.num_jobs):
        for utt, speak in speaker_info[j].items():
            utterance = speaker_classifier.corpus.utterances[utt]
            if speak in speaker_classifier.corpus.speakers:
                speaker = speaker_classifier.corpus.speakers[speak]
            else:
                speaker = Speaker(speak)
                speaker_classifier.corpus.speakers[speak] = speaker
            utterance.set_speaker(speaker)