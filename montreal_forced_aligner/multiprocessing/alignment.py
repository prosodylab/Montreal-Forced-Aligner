from __future__ import annotations
from typing import Collection, TYPE_CHECKING, Optional, Union, List, Dict, Tuple, Set
if TYPE_CHECKING:
    from ..trainers import BaseTrainer, MonophoneTrainer, LdaTrainer, SatTrainer
    from ..aligner.base import BaseAligner
    from ..textgrid import CtmInterval
    from ..aligner.pretrained import PretrainedAligner
    from ..aligner.adapting import AdaptingAligner
    from ..corpus import CorpusType, SegmentsType, OneToOneMappingType, OneToManyMappingType
    from ..corpus.classes import Utterance, File, CleanupWordCtmArguments, NoCleanupWordCtmArguments, PhoneCtmArguments, CombineCtmArguments, \
        CtmToTextGridArguments, ExportTextGridArguments

    from ..dictionary import ReversedMappingType, MultiSpeakerMappingType, MappingType, WordsType, DictionaryType, PunctuationType, IpaType
    from ..config.align_config import AlignConfig, ConfigDict
    ConfigType = Union[BaseTrainer, AlignConfig]

    IterationType = Union[str, int]

    AlignerType = Union[BaseTrainer, BaseAligner]
    CtmType = List[CtmInterval]
import subprocess
import os
import re
import sys
import time
import traceback
import statistics

from ..utils import thirdparty_binary

from .helper import run_mp, run_non_mp

from ..textgrid import parse_from_word, parse_from_word_no_cleanup, parse_from_phone, \
    ctms_to_textgrids_non_mp, output_textgrid_writing_errors, generate_tiers, export_textgrid, process_ctm_line


from ..exceptions import AlignmentError, AlignmentExportError
import multiprocessing as mp
from ..multiprocessing.helper import Stopped
from queue import Empty

CtmErrorDict = Dict[Tuple[str, int], str]

queue_polling_timeout = 1


def acc_stats_func(log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ali_paths: Dict[str, str],
    acc_paths: Dict[str, str],
    model_path: str) -> None:
    model_path = model_path
    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name  in  dictionaries:
            acc_proc = subprocess.Popen([thirdparty_binary('gmm-acc-stats-ali'), model_path,
                                         feature_strings[dict_name],
                                         f"ark,s,cs:{ali_paths[dict_name]}",
                                         acc_paths[dict_name]],
                                        stderr=log_file, env=os.environ)
            acc_proc.communicate()


def acc_stats(aligner: AlignerType):
    """
    Multiprocessing function that computes stats for GMM training

    See http://kaldi-asr.org/doc/gmm-acc-stats-ali_8cc.html for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_mono.sh
    for the bash script this function was extracted from

    Parameters
    ----------
    iteration : int
        Iteration to calculate stats for
    directory : str
        Directory of training (monophone, triphone, speaker-adapted triphone
        training directories)
    split_directory : str
        Directory of training data split into the number of jobs
    num_jobs : int
        The number of processes to use in calculation
    """
    arguments = [j.acc_stats_arguments(aligner) for j in aligner.corpus.jobs]

    if aligner.use_mp:
        run_mp(acc_stats_func, arguments, aligner.working_log_directory)
    else:
        run_non_mp(acc_stats_func, arguments, aligner.working_log_directory)

    log_path = os.path.join(aligner.working_log_directory, f'update.{aligner.iteration}.log')
    with open(log_path, 'w') as log_file:
        acc_files = []
        for a in arguments:
            acc_files.extend(a.acc_paths.values())
        sum_proc = subprocess.Popen([thirdparty_binary('gmm-sum-accs'),
                                     '-'
                                     ] + acc_files, stdout=subprocess.PIPE, stderr=log_file, env=os.environ)
        est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                     f'--write-occs={aligner.next_occs_path}',
                                     f'--mix-up={aligner.current_gaussians}',
                                     f'--power={aligner.power}',
                                     aligner.current_model_path,
                                     "-",
                                     aligner.next_model_path], stdin=sum_proc.stdout,
                                    stderr=log_file, env=os.environ)
        est_proc.communicate()
    if not aligner.debug:
        for f in acc_files:
            os.remove(f)


def compile_train_graphs_func(log_path: str,
    dictionaries: List[str],
    tree_path: str,
    model_path: str,
    text_int_paths: Dict[str, str],
    disambig_paths: Dict[str, str],
    lexicon_fst_paths: Dict[str, str],
    fst_scp_paths: Dict[str, str]) -> None:

    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            disambig_path = disambig_paths[dict_name]
            fst_scp_path = fst_scp_paths[dict_name]
            fst_ark_path = fst_scp_path.replace('.scp', '.ark')
            text_path = text_int_paths[dict_name]
            proc = subprocess.Popen([thirdparty_binary('compile-train-graphs'),
                                     f'--read-disambig-syms={disambig_path}',
                                     tree_path, model_path,
                                     lexicon_fst_paths[dict_name],
                                     f"ark:{text_path}", f"ark,scp:{fst_ark_path},{fst_scp_path}"],
                                    stderr=log_file, env=os.environ)
            proc.communicate()


def compile_train_graphs(
        aligner: AlignerType
) -> None:
    """
    Multiprocessing function that compiles training graphs for utterances

    See http://kaldi-asr.org/doc/compile-train-graphs_8cc.html for more details
    on the Kaldi binary this function calls.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_mono.sh
    for the bash script that this function was extracted from.

    Parameters
    ----------
    directory : str
        Directory of training (monophone, triphone, speaker-adapted triphone
        training directories)
    lang_directory : str
        Directory of the language model used
    split_directory : str
        Directory of training data split into the number of jobs
    num_jobs : int
        The number of processes to use
    """
    aligner.logger.debug('Compiling training graphs...')
    begin = time.time()
    log_directory = aligner.working_log_directory
    os.makedirs(log_directory, exist_ok=True)
    jobs = [x.compile_train_graph_arguments(aligner)
            for x in aligner.corpus.jobs]
    if aligner.use_mp:
        run_mp(compile_train_graphs_func, jobs, log_directory)
    else:
        run_non_mp(compile_train_graphs_func, jobs, log_directory)
    aligner.logger.debug(f'Compiling training graphs took {time.time() - begin}')


def mono_align_equal_func(log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    fst_scp_paths: Dict[str, str],
    ali_ark_paths: Dict[str, str],
    acc_paths: Dict[str, str],
    model_path: str):

    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            fst_path = fst_scp_paths[dict_name]
            ali_path = ali_ark_paths[dict_name]
            acc_path = acc_paths[dict_name]
            align_proc = subprocess.Popen([thirdparty_binary('align-equal-compiled'), f"scp:{fst_path}",
                                           feature_strings[dict_name], f'ark:{ali_path}'],
                                          stderr=log_file, env=os.environ)
            align_proc.communicate()
            stats_proc = subprocess.Popen([thirdparty_binary('gmm-acc-stats-ali'), '--binary=true',
                                           model_path, feature_strings[dict_name], f'ark:{ali_path}', acc_path],
                                          stdin=align_proc.stdout, stderr=log_file, env=os.environ)
            stats_proc.communicate()


def mono_align_equal(aligner: MonophoneTrainer):
    """
    Multiprocessing function that creates equal alignments for base monophone training

    See http://kaldi-asr.org/doc/align-equal-compiled_8cc.html for more details
    on the Kaldi binary this function calls.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_mono.sh
    for the bash script that this function was extracted from.

    Parameters
    ----------
    mono_directory : str
        Directory of monophone training
    split_directory : str
        Directory of training data split into the number of jobs
    num_jobs : int
        The number of processes to use
    """

    arguments = [x.mono_align_equal_arguments(aligner)
            for x in aligner.corpus.jobs]

    if aligner.use_mp:
        run_mp(mono_align_equal_func, arguments, aligner.log_directory)
    else:
        run_non_mp(mono_align_equal_func, arguments, aligner.log_directory)

    log_path = os.path.join(aligner.working_log_directory, 'update.0.log')
    with open(log_path, 'w') as log_file:
        acc_files = []
        for x in arguments:
            acc_files.extend(sorted(x.acc_paths.values()))
        sum_proc = subprocess.Popen([thirdparty_binary('gmm-sum-accs'),
                                     '-']+ acc_files, stderr=log_file, stdout=subprocess.PIPE, env=os.environ)
        est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                     '--min-gaussian-occupancy=3',
                                     f'--mix-up={aligner.current_gaussians}',
                                     f'--power={aligner.power}',
                                     aligner.current_model_path, "-",
                                     aligner.next_model_path],
                                    stderr=log_file, stdin=sum_proc.stdout, env=os.environ)
        est_proc.communicate()
    if not aligner.debug:
        for f in acc_files:
            os.remove(f)


def align_func(
        log_path: str,
    dictionaries: List[str],
    fst_scp_paths: Dict[str, str],
feature_strings: Dict[str, str],
model_path: str,
ali_paths: Dict[str, str],
score_paths: Dict[str, str],
loglike_paths: Dict[str, str],
align_options: ConfigDict
):
    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            fst_path = fst_scp_paths[dict_name]
            ali_path = ali_paths[dict_name]
            com = [thirdparty_binary('gmm-align-compiled'),
                   f"--transition-scale={align_options['transition_scale']}",
                   f"--acoustic-scale={align_options['acoustic_scale']}",
                   f"--self-loop-scale={align_options['self_loop_scale']}",
                   f"--beam={align_options['beam']}",
                   f"--retry-beam={align_options['retry_beam']}",
                   '--careful=false',
                   '-',
                   f"scp:{fst_path}", feature_string, f"ark:{ali_path}"]
            if align_options['debug']:
                loglike_path = loglike_paths[dict_name]
                score_path = score_paths[dict_name]
                com.insert(1, f"--write-per-frame-acoustic-loglikes=ark,t:{loglike_path}")
                com.append(f"ark,t:{score_path}")

            boost_proc = subprocess.Popen([thirdparty_binary('gmm-boost-silence'),
                                           f"--boost={align_options['boost_silence']}",
                                           align_options['optional_silence_csl'],
                                           model_path,
                                           '-'], stderr= log_file, stdout=subprocess.PIPE, env=os.environ)
            align_proc = subprocess.Popen(com,
                                          stderr=log_file, stdin=boost_proc.stdout, env=os.environ)
            align_proc.communicate()


def align(
        aligner: AlignerType
) -> None:
    """
    Multiprocessing function that aligns based on the current model

    See http://kaldi-asr.org/doc/gmm-align-compiled_8cc.html and
    http://kaldi-asr.org/doc/gmm-boost-silence_8cc.html for more details
    on the Kaldi binary this function calls.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/align_si.sh
    for the bash script this function was based on.

    Parameters
    ----------
    iteration : int or str
        Iteration to align
    directory : str
        Directory of training (monophone, triphone, speaker-adapted triphone
        training directories)
    split_directory : str
        Directory of training data split into the number of jobs
    optional_silence : str
        Colon-separated list of silence phones to boost
    num_jobs : int
        The number of processes to use in calculation
    config : :class:`~aligner.config.MonophoneConfig`, :class:`~aligner.config.TriphoneConfig` or :class:`~aligner.config.TriphoneFmllrConfig`
        Configuration object for training
    """
    begin = time.time()
    log_directory = aligner.working_log_directory

    arguments = [x.align_arguments(aligner) for x in aligner.corpus.jobs]
    if aligner.use_mp:
        run_mp(align_func, arguments, log_directory)
    else:
        run_non_mp(align_func, arguments, log_directory)

    error_logs = []
    for j in arguments:

        with open(j.log_path, 'r', encoding='utf8') as f:
            for line in f:
                if line.strip().startswith('ERROR'):
                    error_logs.append(j.log_path)
                    break
    if error_logs:
        raise AlignmentError(error_logs)
    aligner.logger.debug(f'Alignment round took {time.time() - begin}')


def compile_information_func(
    align_log_path: str
) -> Dict[str, Union[List[str], float, int]]:

    log_like_pattern = re.compile(
        r'^LOG .* Overall log-likelihood per frame is (?P<log_like>[-0-9.]+) over (?P<frames>\d+) frames.*$')

    decode_error_pattern = re.compile(r'^WARNING .* Did not successfully decode file (?P<utt>.*?), .*$')

    data = {'unaligned': [], 'too_short': [], 'log_like': 0, 'total_frames': 0}
    with open(align_log_path, 'r', encoding='utf8') as f:
        for line in f:
            decode_error_match = re.match(decode_error_pattern, line)
            if decode_error_match:
                data['unaligned'].append(decode_error_match.group('utt'))
                continue
            log_like_match = re.match(log_like_pattern, line)
            if log_like_match:
                log_like = log_like_match.group('log_like')
                frames = log_like_match.group('frames')
                data['log_like'] = float(log_like)
                data['total_frames'] = int(frames)
    return data


def compile_information(aligner: AlignerType) -> Tuple[Dict[str, str], float]:
    compile_info_begin = time.time()

    jobs = [x.compile_information_arguments(aligner)
            for x in aligner.corpus.jobs]

    if aligner.use_mp:
        manager = mp.Manager()
        alignment_info = manager.dict()
        run_mp(compile_information_func, jobs, aligner.working_log_directory, alignment_info)
    else:
        alignment_info = {}
        alignment_info = run_non_mp(compile_information_func, jobs, aligner.working_log_directory, alignment_info)

    unaligned = {}
    total_frames = sum(data['total_frames'] for data in alignment_info.values())
    average_log_like = 0
    for x, data in alignment_info.items():
        if total_frames:
            weight = data['total_frames'] / total_frames
            average_log_like += data['log_like'] * weight
        for u in data['unaligned']:
            unaligned[u] = 'Beam too narrow'
        for u in data['too_short']:
            unaligned[u] = 'Segment too short'

    if not total_frames:
        aligner.logger.warning('No files were aligned, this likely indicates serious problems with the aligner.')
    aligner.logger.debug(f'Compiling information took {time.time() - compile_info_begin}')
    return unaligned, average_log_like


def compute_alignment_improvement_func(
    log_path: str,
    dictionaries: List[str],
    model_path: str,
    text_int_paths: Dict[str, str],
    word_boundary_paths: Dict[str, str],
    ali_paths: Dict[str, str],
    frame_shift: int,
    reversed_phone_mappings: Dict[str, Dict[int, str]],
    positions: Dict[str, List[str]],
    phone_ctm_paths: Dict[str, str]
) -> None:
    try:

        frame_shift = frame_shift / 1000
        with open(log_path, 'w', encoding='utf8') as log_file:
            for dict_name in dictionaries:
                text_int_path = text_int_paths[dict_name]
                ali_path = ali_paths[dict_name]
                phone_ctm_path = phone_ctm_paths[dict_name]
                word_boundary_path = word_boundary_paths[dict_name]
                if os.path.exists(phone_ctm_path):
                    continue

                lin_proc = subprocess.Popen([thirdparty_binary('linear-to-nbest'), f"ark:{ali_path}",
                                             f"ark:{text_int_path}",
                                             '', '', 'ark:-'],
                                            stdout=subprocess.PIPE, stderr=log_file, env=os.environ)
                det_proc = subprocess.Popen([thirdparty_binary('lattice-determinize-pruned'),
                                             'ark:-', 'ark:-'],
                                            stdin=lin_proc.stdout, stderr=log_file,
                                            stdout=subprocess.PIPE, env=os.environ)
                align_proc = subprocess.Popen([thirdparty_binary('lattice-align-words'),
                                               word_boundary_path, model_path,
                                               'ark:-', 'ark:-'],
                                              stdin=det_proc.stdout, stderr=log_file,
                                              stdout=subprocess.PIPE, env=os.environ)
                phone_proc = subprocess.Popen([thirdparty_binary('lattice-to-phone-lattice'), model_path,
                                               'ark:-', "ark:-"],
                                              stdin=align_proc.stdout,
                                              stdout=subprocess.PIPE,
                                              stderr=log_file, env=os.environ)
                nbest_proc = subprocess.Popen([thirdparty_binary('nbest-to-ctm'),
                                               f'--frame-shift={frame_shift}',
                                               "ark:-", phone_ctm_path],
                                              stdin=phone_proc.stdout,
                                              stderr=log_file, env=os.environ)
                nbest_proc.communicate()
            mapping = reversed_phone_mappings[dict_name]
            actual_lines = []
            with open(phone_ctm_path, 'r', encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    if line == '':
                        continue
                    line = line.split(' ')
                    utt = line[0]
                    begin = float(line[2])
                    duration = float(line[3])
                    end = begin + duration
                    label = line[4]
                    try:
                        label = mapping[int(label)]
                    except KeyError:
                        pass
                    for p in positions[dict_name]:
                        if label.endswith(p):
                            label = label[:-1 * len(p)]
                    actual_lines.append([utt, begin, end, label])
            with open(phone_ctm_path, 'w', encoding='utf8') as f:
                for line in actual_lines:
                    f.write(f"{' '.join(map(str, line))}\n")
    except Exception as e:
        raise (Exception(str(e)))


def parse_iteration_alignments(
        aligner: AlignerType,
        iteration: Optional[IterationType]=None
) -> Dict[str, List[Tuple[float, float, str]]]:
    if iteration is None:
        iteration = aligner.iteration
    data = {}
    for j in aligner.corpus.jobs:
        phone_ctm_path = os.path.join(aligner.working_directory, f'phone.{iteration}.{j.name}.ctm')
        with open(phone_ctm_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                line = line.split(' ')
                utt = line[0]
                begin = float(line[1])
                end = float(line[2])
                label = line[3]
                if utt not in data:
                    data[utt] = []
                data[utt].append((begin, end, label))
    return data


def compare_alignments(
        alignments_one: Dict[str, List[Tuple[float, float, str]]],
        alignments_two: Dict[str, List[Tuple[float, float, str]]],
        frame_shift: int
):
    utterances_aligned_diff = len(alignments_two) - len(alignments_one)
    utts_one = set(alignments_one.keys())
    utts_two = set(alignments_two.keys())
    common_utts = utts_one.intersection(utts_two)
    differences = []
    for u in common_utts:
        end = alignments_one[u][-1][1]
        t = 0
        one_alignment = alignments_one[u]
        two_alignment = alignments_two[u]
        difference = 0
        while t < end:
            one_label = None
            two_label = None
            for b, e, l in one_alignment:
                if t < b:
                    continue
                if t >= e:
                    break
                one_label = l
            for b, e, l in two_alignment:
                if t < b:
                    continue
                if t >= e:
                    break
                two_label = l
            if one_label != two_label:
                difference += frame_shift
            t += frame_shift
        difference /= end
        differences.append(difference)
    if differences:
        mean_difference = statistics.mean(differences)
    else:
        mean_difference = 'N/A'
    return utterances_aligned_diff, mean_difference


def compute_alignment_improvement(
        aligner: AlignerType
) -> None:
    jobs = [x.alignment_improvement_arguments(aligner)
            for x in aligner.corpus.jobs]
    if aligner.use_mp:
        run_mp(compute_alignment_improvement_func, jobs, aligner.working_log_directory)
    else:
        run_non_mp(compute_alignment_improvement_func, jobs, aligner.working_log_directory)

    alignment_diff_path = os.path.join(aligner.working_directory, 'train_change.csv')
    if aligner.iteration == 0 or aligner.iteration not in aligner.realignment_iterations:
        return
    ind = aligner.realignment_iterations.index(aligner.iteration)
    if ind != 0:
        previous_iteration = aligner.realignment_iterations[ind - 1]
    else:
        previous_iteration = 0
    try:
        previous_alignments = parse_iteration_alignments(aligner, previous_iteration)
    except FileNotFoundError:
        return
    current_alignments = parse_iteration_alignments(aligner)
    utterance_aligned_diff, mean_difference = compare_alignments(previous_alignments, current_alignments,
                                                                 aligner.feature_config.frame_shift)
    if not os.path.exists(alignment_diff_path):
        with open(alignment_diff_path, 'w', encoding='utf8') as f:
            f.write('iteration,number_aligned,number_previously_aligned,'
                    'difference_in_utts_aligned,mean_boundary_change\n')
    if aligner.iteration in aligner.realignment_iterations:
        with open(alignment_diff_path, 'a', encoding='utf8') as f:
            f.write(f'{aligner.iteration},{len(current_alignments)},{len(previous_alignments)},'
                    f'{utterance_aligned_diff},{mean_difference}\n')
    if not aligner.debug:
        for j in jobs:
            for p in j.phone_ctm_paths:
                os.remove(p)


def ali_to_ctm_func(
    log_path: str,
    dictionaries: List[str],
    ali_paths: Dict[str, str],
    text_int_paths: Dict[str, str],
    word_boundary_int_paths: Dict[str, str],
    frame_shift: float,
    model_path: str,
    ctm_paths: Dict[str, str],
    word_mode: bool
) -> None:

    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            ali_path = ali_paths[dict_name]
            text_int_path = text_int_paths[dict_name]
            ctm_path = ctm_paths[dict_name]
            word_boundary_int_path = word_boundary_int_paths[dict_name]
            if os.path.exists(ctm_path):
                return
            lin_proc = subprocess.Popen([thirdparty_binary('linear-to-nbest'), "ark:" + ali_path,
                                         "ark:" + text_int_path,
                                         '', '', 'ark:-'],
                                        stdout=subprocess.PIPE, stderr=log_file, env=os.environ)
            align_words_proc = subprocess.Popen([thirdparty_binary('lattice-align-words'),
                                                 word_boundary_int_path, model_path,
                                                 'ark:-', 'ark:-'],
                                                stdin=lin_proc.stdout, stdout=subprocess.PIPE,
                                                stderr=log_file, env=os.environ)
            if word_mode:
                nbest_proc = subprocess.Popen([thirdparty_binary('nbest-to-ctm'),
                                               f'--frame-shift={frame_shift}',
                                               'ark:-',
                                               ctm_path],
                                              stderr=log_file, stdin=align_words_proc.stdout, env=os.environ)
            else:
                phone_proc = subprocess.Popen([thirdparty_binary('lattice-to-phone-lattice'), model_path,
                                               'ark:-', "ark:-"],
                                              stdout=subprocess.PIPE, stdin=align_words_proc.stdout,
                                              stderr=log_file, env=os.environ)
                nbest_proc = subprocess.Popen([thirdparty_binary('nbest-to-ctm'),
                                               f'--frame-shift={frame_shift}',
                                               "ark:-", ctm_path],
                                              stdin=phone_proc.stdout,
                                              stderr=log_file, env=os.environ)
            nbest_proc.communicate()


class NoCleanupWordCtmProcessWorker(mp.Process):
    def __init__(self,
                 job_name: int,
                 to_process_queue: mp.Queue,
                 stopped: Stopped,
                 error_catching: CtmErrorDict,
                 arguments: NoCleanupWordCtmArguments
                 ):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.dictionaries = arguments.dictionaries
        self.ctm_paths = arguments.ctm_paths
        self.to_process_queue = to_process_queue
        self.stopped = stopped
        self.error_catching = error_catching

        # Corpus information
        self.utterances = arguments.utterances

        # Dictionary information
        self.dictionary_data = arguments.dictionary_data

    def run(self) -> None:
        current_file_data = {}

        def process_current(cur_utt : Utterance, current_labels: CtmType):
            actual_labels = parse_from_word_no_cleanup(current_labels, self.dictionary_data[dict_name].reversed_words_mapping)
            current_file_data[cur_utt.name] = actual_labels

        def process_current_file(cur_file: str):
            self.to_process_queue.put(('word', dict_name, cur_file, current_file_data))

        cur_utt = None
        cur_file = None
        utt_begin = 0
        current_labels = []
        try:
            for dict_name in self.dictionaries:
                with open(self.ctm_paths[dict_name], 'r') as word_file:
                    for line in word_file:
                        line = line.strip()
                        if not line:
                            continue
                        interval = process_ctm_line(line)
                        utt = interval.utterance
                        if cur_utt is None:
                            cur_utt = self.utterances[dict_name][utt]
                            utt_begin = cur_utt.begin
                            cur_file = cur_utt.file_name

                        if utt != cur_utt:
                            process_current(cur_utt, current_labels)
                            cur_utt = self.utterances[dict_name][utt]
                            file_name = cur_utt.file_name
                            if file_name != cur_file:
                                process_current_file(cur_file)
                                current_file_data = {}
                                cur_file = file_name
                            current_labels = []
                        if utt_begin:
                            interval.shift_times(utt_begin)
                        current_labels.append(interval)
                if current_labels:
                    process_current(cur_utt, current_labels)
                    process_current_file(cur_file)
        except Exception as e:
            self.stopped.stop()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.error_catching[('word', self.job_name)] = '\n'.join(
                traceback.format_exception(exc_type, exc_value, exc_traceback))


class CleanupWordCtmProcessWorker(mp.Process):
    def __init__(self, job_name: int, to_process_queue: mp.Queue,
                 stopped: Stopped, error_catching: CtmErrorDict,
                 arguments: CleanupWordCtmArguments):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.dictionaries = arguments.dictionaries
        self.ctm_paths = arguments.ctm_paths
        self.to_process_queue = to_process_queue
        self.stopped = stopped
        self.error_catching = error_catching

        # Corpus information
        self.utterances = arguments.utterances

        # Dictionary information
        self.dictionary_data = arguments.dictionary_data

    def run(self) -> None:
        current_file_data = {}

        def process_current(cur_utt: Utterance, current_labels: CtmType) -> None:
            text = cur_utt.text.split()
            actual_labels = parse_from_word(current_labels, text, self.dictionary_data[dict_name])

            current_file_data[cur_utt.name] = actual_labels

        def process_current_file(cur_file: str) -> None:
            self.to_process_queue.put(('word', dict_name, cur_file, current_file_data))

        cur_utt = None
        cur_file = None
        utt_begin = 0
        current_labels = []
        try:
            for dict_name in self.dictionaries:
                ctm_path = self.ctm_paths[dict_name]
                with open(ctm_path, 'r') as word_file:
                    for line in word_file:
                        line = line.strip()
                        if not line:
                            continue
                        interval = process_ctm_line(line)
                        utt = interval.utterance
                        if cur_utt is None:
                            cur_utt = self.utterances[dict_name][utt]
                            utt_begin = cur_utt.begin
                            cur_file = cur_utt.file_name

                        if utt != cur_utt:
                            process_current(cur_utt, current_labels)
                            cur_utt = self.utterances[dict_name][utt]
                            utt_begin = cur_utt.begin
                            file_name = cur_utt.file_name
                            if file_name != cur_file:
                                process_current_file(cur_file)
                                current_file_data = {}
                                cur_file = file_name
                            current_labels = []
                        if utt_begin:
                            interval.shift_times(utt_begin)
                        current_labels.append(interval)
                if current_labels:
                    process_current(cur_utt, current_labels)
                    process_current_file(cur_file)
        except Exception as e:
            self.stopped.stop()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.error_catching[('word', self.job_name)] = '\n'.join(
                traceback.format_exception(exc_type, exc_value, exc_traceback))


class PhoneCtmProcessWorker(mp.Process):
    def __init__(self, job_name: int, to_process_queue: mp.Queue,
                 stopped: Stopped, error_catching: CtmErrorDict,
                 arguments: PhoneCtmArguments):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.dictionaries = arguments.dictionaries
        self.ctm_paths = arguments.ctm_paths
        self.to_process_queue = to_process_queue
        self.stopped = stopped
        self.error_catching = error_catching

        self.utterances = arguments.utterances

        self.reversed_phone_mappings = arguments.reversed_phone_mappings
        self.positions = arguments.positions

    def run(self) -> None:
        cur_utt = None
        cur_file = None
        utt_begin = 0
        current_labels = []

        current_file_data = {}

        def process_current_utt(cur_utt: Utterance, current_labels: CtmType) -> None:
            actual_labels = parse_from_phone(current_labels, self.reversed_phone_mappings[dict_name], self.positions[dict_name])
            current_file_data[cur_utt.name] = actual_labels

        def process_current_file(cur_file: str) -> None:
            self.to_process_queue.put(('phone', dict_name, cur_file, current_file_data))

        try:
            for dict_name in self.dictionaries:
                with open(self.ctm_paths[dict_name], 'r') as word_file:
                    for line in word_file:
                        line = line.strip()
                        if not line:
                            continue
                        interval = process_ctm_line(line)
                        utt = interval.utterance
                        if cur_utt is None:
                            cur_utt = self.utterances[dict_name][utt]
                            cur_file = cur_utt.file_name
                            utt_begin = cur_utt.begin

                        if utt != cur_utt:

                            process_current_utt(cur_utt, current_labels)

                            cur_utt = self.utterances[dict_name][utt]
                            file_name = cur_utt.file_name
                            utt_begin = cur_utt.begin

                            if file_name != cur_file:
                                process_current_file(cur_file)
                                current_file_data = {}
                                cur_file = file_name
                            current_labels = []
                        if utt_begin:
                            interval.shift_times(utt_begin)
                        current_labels.append(interval)
                if current_labels:
                    process_current_utt(cur_utt, current_labels)
                    process_current_file(cur_file)
        except Exception as e:
            self.stopped.stop()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.error_catching[('phone', self.job_name)] = '\n'.join(
                traceback.format_exception(exc_type, exc_value, exc_traceback))


class CombineProcessWorker(mp.Process):
    def __init__(self, job_name: int, to_process_queue: mp.Queue, to_export_queue: mp.Queue,
                 stopped: Stopped, finished_combining: Stopped, error_catching: CtmErrorDict,
                 arguments: CombineCtmArguments):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.to_process_queue = to_process_queue
        self.to_export_queue = to_export_queue
        self.stopped = stopped
        self.finished_combining = finished_combining
        self.error_catching = error_catching

        self.files = arguments.files
        self.dictionary_data = arguments.dictionary_data
        self.cleanup_textgrids = arguments.cleanup_textgrids

    def run(self) -> None:
        sum_time = 0
        count_time = 0
        phone_data = {}
        word_data = {}
        while True:
            try:
                w_p, dict_name, file_name, data = self.to_process_queue.get(timeout=queue_polling_timeout)
                begin_time = time.time()
            except Empty as error:
                if self.finished_combining.stop_check():
                    break
                continue
            self.to_process_queue.task_done()
            if self.stopped.stop_check():
                continue
            if w_p == 'phone':
                if file_name in word_data:
                    word_ctm = word_data.pop(file_name)
                    phone_ctm = data
                else:
                    phone_data[file_name] = data
                    continue
            else:
                if file_name in phone_data:
                    phone_ctm = phone_data.pop(file_name)
                    word_ctm = data
                else:
                    word_data[file_name] = data
                    continue
            try:
                from ..corpus.classes import File
                file = self.files[file_name]
                for u_name, u in file.utterances.items():
                    if u_name not in word_ctm:
                        continue
                    u.word_labels = word_ctm[u_name]
                    u.phone_labels = phone_ctm[u_name]
                data = generate_tiers(file, cleanup_textgrids=self.cleanup_textgrids)
                self.to_export_queue.put((file_name, data))
            except Exception as e:
                self.stopped.stop()
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.error_catching[('combining', self.job_name)] = '\n'.join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback))

            sum_time += time.time() - begin_time
            count_time += 1


class ExportTextGridProcessWorker(mp.Process):
    def __init__(self, for_write_queue: mp.Queue, stopped: Stopped, finished_processing: Stopped,
                 textgrid_errors: Dict[str, str],
                 arguments: ExportTextGridArguments):
        mp.Process.__init__(self)
        self.for_write_queue = for_write_queue
        self.stopped = stopped
        self.finished_processing = finished_processing
        self.textgrid_errors = textgrid_errors

        self.files = arguments.files
        self.output_directory = arguments.output_directory
        self.backup_output_directory = arguments.backup_output_directory

        self.frame_shift = arguments.frame_shift

    def run(self) -> None:
        while True:
            try:
                file_name, data = self.for_write_queue.get(timeout=queue_polling_timeout)
            except Empty as error:
                if self.finished_processing.stop_check():
                    break
                continue
            self.for_write_queue.task_done()
            if self.stopped.stop_check():
                continue
            try:
                overwrite = True
                file = self.files[file_name]
                output_path = file.construct_output_path(self.output_directory, self.backup_output_directory)

                export_textgrid(file, output_path, data,
                                self.frame_shift, overwrite)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.textgrid_errors[file_name] = '\n'.join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback))


class ExportPreparationProcessWorker(mp.Process):
    def __init__(self, to_export_queue: mp.Queue, for_write_queue: mp.Queue,
                 stopped: Stopped, finished_combining: Stopped, files: Dict[str, File]):
        mp.Process.__init__(self)
        self.to_export_queue = to_export_queue
        self.for_write_queue = for_write_queue
        self.stopped = stopped
        self.finished_combining = finished_combining

        self.files = files


    def run(self) -> None:
        export_data = {}
        try:
            while True:
                try:
                    file_name, data = self.to_export_queue.get(timeout=queue_polling_timeout)
                except Empty as error:
                    if self.finished_combining.stop_check():
                        break
                    continue
                self.to_export_queue.task_done()
                if self.stopped.stop_check():
                    continue
                file = self.files[file_name]
                if len(file.speaker_ordering) > 1:
                    if file_name not in export_data:
                        export_data[file_name] = data
                    else:
                        export_data[file_name].update(data)
                    if len(export_data[file_name]) == len(file.speaker_ordering):
                        data = export_data.pop(file_name)
                        self.for_write_queue.put((file_name, data))
                else:
                    self.for_write_queue.put((file_name, data))

            for k, v in export_data.items():
                self.for_write_queue.put((k, v))
        except Exception:
            self.stopped.stop()
            raise

def ctms_to_textgrids_mp(aligner: AlignerType):
    export_begin = time.time()
    manager = mp.Manager()
    textgrid_errors = manager.dict()
    error_catching = manager.dict()
    stopped = Stopped()
    backup_output_directory = None
    if not aligner.align_config.overwrite:
        backup_output_directory = os.path.join(aligner.align_directory, 'textgrids')
        os.makedirs(backup_output_directory, exist_ok=True)


    aligner.logger.debug('Beginning to process ctm files...')
    ctm_begin_time = time.time()
    word_procs = []
    phone_procs = []
    combine_procs = []
    finished_signals = [Stopped() for _ in range(aligner.corpus.num_jobs)]
    finished_processing = Stopped()
    to_process_queue = [mp.JoinableQueue() for _ in range(aligner.corpus.num_jobs)]
    to_export_queue = mp.JoinableQueue()
    for_write_queue = mp.JoinableQueue()
    finished_combining = Stopped()
    for j in aligner.corpus.jobs:
        if aligner.align_config.cleanup_textgrids:
            word_p = CleanupWordCtmProcessWorker(j.name, to_process_queue[j.name], stopped, error_catching,
                                                 j.cleanup_word_ctm_arguments(aligner))
        else:
            word_p = NoCleanupWordCtmProcessWorker(j.name, to_process_queue[j.name], stopped, error_catching,
                                                   j.no_cleanup_word_ctm_arguments(aligner))

        word_procs.append(word_p)
        word_p.start()

        phone_p = PhoneCtmProcessWorker(j.name, to_process_queue[j.name], stopped, error_catching,
                                        j.phone_ctm_arguments(aligner))
        phone_p.start()
        phone_procs.append(phone_p)

        combine_p = CombineProcessWorker(j.name, to_process_queue[j.name], to_export_queue, stopped, finished_signals[j.name],
                                         error_catching,
                                         j.combine_ctm_arguments(aligner))
        combine_p.start()
        combine_procs.append(combine_p)
    preparation_proc = ExportPreparationProcessWorker(to_export_queue, for_write_queue, stopped, finished_combining,
                                                      aligner.corpus.files)
    preparation_proc.start()

    export_procs = []
    for j in aligner.corpus.jobs:
        export_proc = ExportTextGridProcessWorker(for_write_queue, stopped, finished_processing, textgrid_errors,
                                                  j.export_textgrid_arguments(aligner))
        export_proc.start()
        export_procs.append(export_proc)

    aligner.logger.debug('Waiting for processes to finish...')
    for i in range(aligner.corpus.num_jobs):
        word_procs[i].join()
        phone_procs[i].join()
        finished_signals[i].stop()

    aligner.logger.debug(f'Ctm parsers took {time.time() - ctm_begin_time} seconds')

    aligner.logger.debug('Waiting for processes to finish...')
    for i in range(aligner.corpus.num_jobs):
        to_process_queue[i].join()
        combine_procs[i].join()
    finished_combining.stop()

    to_export_queue.join()
    preparation_proc.join()

    aligner.logger.debug(f'Combiners took {time.time() - ctm_begin_time} seconds')
    aligner.logger.debug('Beginning export...')

    aligner.logger.debug(f'Adding jobs for export took {time.time() - export_begin}')
    aligner.logger.debug('Waiting for export processes to join...')

    for_write_queue.join()
    finished_processing.stop()
    for i in range(aligner.corpus.num_jobs):
        export_procs[i].join()
    for_write_queue.join()
    aligner.logger.debug(f'Export took {time.time() - export_begin} seconds')

    if error_catching:
        aligner.logger.error('Error was encountered in processing CTMs')
        for key, error in error_catching.items():
            aligner.logger.error(f'{key}:\n\n{error}')
        raise AlignmentExportError(error_catching)

    output_textgrid_writing_errors(aligner.textgrid_output, textgrid_errors)


def convert_ali_to_textgrids(aligner: AlignerType) -> None:
    """
    Multiprocessing function that aligns based on the current model

    See:

    - http://kaldi-asr.org/doc/linear-to-nbest_8cc.html
    - http://kaldi-asr.org/doc/lattice-align-words_8cc.html
    - http://kaldi-asr.org/doc/lattice-to-phone-lattice_8cc.html
    - http://kaldi-asr.org/doc/nbest-to-ctm_8cc.html

    for more details
    on the Kaldi binaries this function calls.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/get_train_ctm.sh
    for the bash script that this function was based on.

    Parameters
    ----------
    output_directory : str
        Directory to write TextGrid files to
    model_directory : str
        Directory of training (monophone, triphone, speaker-adapted triphone
        training directories)
    dictionary : :class:`~montreal_forced_aligner.dictionary.Dictionary`
        Dictionary object that has information about pronunciations
    corpus : :class:`~montreal_forced_aligner.corpus.AlignableCorpus`
        Corpus object that has information about the dataset
    num_jobs : int
        The number of processes to use in calculation

    Raises
    ------
    CorpusError
        If the files per speaker exceeds the number of files that are
        allowed to be open on the computer (for Unix-based systems)

    """
    log_directory = aligner.working_log_directory
    os.makedirs(aligner.textgrid_output, exist_ok=True)
    jobs = [x.ali_to_word_ctm_arguments(aligner)  # Word CTM jobs
            for x in aligner.corpus.jobs]
    jobs += [x.ali_to_phone_ctm_arguments(aligner)  # Phone CTM jobs
             for x in aligner.corpus.jobs]
    aligner.logger.info('Generating CTMs from alignment...')
    if aligner.use_mp:
        run_mp(ali_to_ctm_func, jobs, log_directory)
    else:
        run_non_mp(ali_to_ctm_func, jobs, log_directory)
    aligner.logger.info('Finished generating CTMs!')

    aligner.logger.info('Exporting TextGrids from CTMs...')
    if aligner.use_mp:
        ctms_to_textgrids_mp(aligner)
    else:
        ctms_to_textgrids_non_mp(aligner)
    aligner.logger.info('Finished exporting TextGrids!')


def tree_stats_func(log_path: str,
    dictionaries: List[str],
    ci_phones: str,
    model_path: str,
    feature_strings: Dict[str, str],
    ali_paths: Dict[str, str],
    treeacc_paths: Dict[str, str]) -> None:


    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            ali_path = ali_paths[dict_name]
            treeacc_path = treeacc_paths[dict_name]
            subprocess.call([thirdparty_binary('acc-tree-stats'),
                             f'--ci-phones={ci_phones}', model_path, feature_string,
                             f"ark:{ali_path}",
                             treeacc_path], stderr=log_file)


def tree_stats(aligner: AlignerType) -> None:
    """
    Multiprocessing function that computes stats for decision tree training

    See http://kaldi-asr.org/doc/acc-tree-stats_8cc.html for more details
    on the Kaldi binary this runs.

    Parameters
    ----------
    directory : str
        Directory of training (triphone, speaker-adapted triphone
        training directories)
    align_directory : str
        Directory of previous alignment
    split_directory : str
        Directory of training data split into the number of jobs
    ci_phones : str
        Colon-separated list of context-independent phones
    num_jobs : int
        The number of processes to use in calculation
    """

    jobs = [j.tree_stats_arguments(aligner)
            for j in aligner.corpus.jobs]

    if aligner.use_mp:
        run_mp(tree_stats_func, jobs, aligner.working_log_directory)
    else:
        run_non_mp(tree_stats_func, jobs, aligner.working_log_directory)

    tree_accs = []
    for x in jobs:
        tree_accs.extend(x.treeacc_paths.values())
    log_path = os.path.join(aligner.working_log_directory, 'sum_tree_acc.log')
    with open(log_path, 'w', encoding='utf8') as log_file:
        subprocess.call([thirdparty_binary('sum-tree-stats'), os.path.join(aligner.working_directory, 'treeacc')] +
                        tree_accs, stderr=log_file)
    if not aligner.debug:
         for f in tree_accs:
            os.remove(f)


def convert_alignments_func(
    log_path: str,
    dictionaries: List[str],
    model_path: str,
    tree_path: str,
    align_model_path: str,
    ali_paths: Dict[str, str],
    new_ali_paths: Dict[str, str]) -> None:
    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            ali_path = ali_paths[dict_name]
            new_ali_path = new_ali_paths[dict_name]
            subprocess.call([thirdparty_binary('convert-ali'), align_model_path,
                             model_path, tree_path, f"ark:{ali_path}",
                             f"ark:{new_ali_path}"], stderr=log_file)


def convert_alignments(aligner: AlignerType) -> None:
    """
    Multiprocessing function that converts alignments from previous training

    See http://kaldi-asr.org/doc/convert-ali_8cc.html for more details
    on the Kaldi binary this runs.

    Parameters
    ----------
    directory : str
        Directory of training (triphone, speaker-adapted triphone
        training directories)
    align_directory : str
        Directory of previous alignment
    num_jobs : int
        The number of processes to use in calculation

    """

    jobs = [x.convert_alignment_arguments(aligner)
            for x in aligner.corpus.jobs]
    if aligner.use_mp:
        run_mp(convert_alignments_func, jobs, aligner.working_log_directory)
    else:
        run_non_mp(convert_alignments_func, jobs, aligner.working_log_directory)


def calc_fmllr_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ali_paths: Dict[str, str],
    model_path: str,
    spk2utt_paths: Dict[str, str],
    trans_paths: Dict[str, str],
    fmllr_options: ConfigDict) -> None:
    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            ali_path = ali_paths[dict_name]
            spk2utt_path = spk2utt_paths[dict_name]
            trans_path = trans_paths[dict_name]
            post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                          f"ark:{ali_path}", 'ark:-'],
                                         stderr=log_file, stdout=subprocess.PIPE, env=os.environ)

            weight_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'), '0.0',
                                            fmllr_options['silence_csl'], model_path, 'ark:-',
                                            'ark:-'],
                                           stderr=log_file, stdin=post_proc.stdout,
                                           stdout=subprocess.PIPE, env=os.environ)

            if not fmllr_options['initial']:
                cmp_trans_path = trans_paths[dict_name] + '.tmp'
                est_proc = subprocess.Popen([thirdparty_binary('gmm-est-fmllr'),
                                             '--verbose=4',
                                             f"--fmllr-update-type={fmllr_options['fmllr_update_type']}",
                                             f'--spk2utt=ark:{spk2utt_path}', model_path, feature_string,
                                             'ark:-', 'ark:-'],
                                            stderr=log_file, stdin=weight_proc.stdout,
                                            stdout=subprocess.PIPE, env=os.environ)
                comp_proc = subprocess.Popen([thirdparty_binary('compose-transforms'),
                                              '--b-is-affine=true',
                                              'ark:-', f'ark:{trans_path}',
                                              f'ark:{cmp_trans_path}'], stderr=log_file,
                                             stdin=est_proc.stdout, env=os.environ)
                comp_proc.communicate()

                os.remove(trans_path)
                os.rename(cmp_trans_path, trans_path)
            else:
                est_proc = subprocess.Popen([thirdparty_binary('gmm-est-fmllr'),
                                             '--verbose=4',
                                             f"--fmllr-update-type={fmllr_options['fmllr_update_type']}",
                                             f'--spk2utt=ark:{spk2utt_path}', model_path, feature_string,
                                             'ark,s,cs:-', f'ark:{trans_path}'],
                                            stderr=log_file, stdin=weight_proc.stdout, env=os.environ)
                est_proc.communicate()


def calc_fmllr(aligner: AlignerType) -> None:
    """
    Multiprocessing function that computes speaker adaptation (fMLLR)

    See:

    - http://kaldi-asr.org/doc/gmm-est-fmllr_8cc.html
    - http://kaldi-asr.org/doc/ali-to-post_8cc.html
    - http://kaldi-asr.org/doc/weight-silence-post_8cc.html
    - http://kaldi-asr.org/doc/compose-transforms_8cc.html
    - http://kaldi-asr.org/doc/transform-feats_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/align_fmllr.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    directory : str
        Directory of training (triphone, speaker-adapted triphone
        training directories)
    split_directory : str
        Directory of training data split into the number of jobs
    silence_csl : str
        Colon-separated list of silence phones
    num_jobs : int
        The number of processes to use in calculation
    config : :class:`~aligner.config.TriphoneFmllrConfig`
        Configuration object for training
    initial : bool, optional
        Whether this is the first computation of speaker-adaptation,
        defaults to False
    iteration : int or str
        Specifies the current iteration, defaults to None

    """
    begin = time.time()
    log_directory = aligner.working_log_directory

    jobs = [x.calc_fmllr_arguments(aligner) for x in aligner.corpus.jobs]
    if aligner.use_mp:
        run_mp(calc_fmllr_func, jobs, log_directory)
    else:
        run_non_mp(calc_fmllr_func, jobs, log_directory)
    aligner.speaker_independent = False
    aligner.logger.debug(f'Fmllr calculation took {time.time() - begin}')


def acc_stats_two_feats_func(log_path: str,
    dictionaries: List[str],
    ali_paths: Dict[str, str],
    acc_paths: Dict[str, str],
    model_path: str,
    feature_strings: Dict[str, str],
    si_feature_strings: Dict[str, str]) -> None:
    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            ali_path = ali_paths[dict_name]
            acc_path = acc_paths[dict_name]
            feature_string = feature_strings[dict_name]
            si_feature_string = si_feature_strings[dict_name]
            ali_to_post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                                 f'ark:{ali_path}',
                                                 'ark:-'],
                                                stderr=log_file, stdout=subprocess.PIPE, env=os.environ)
            acc_proc = subprocess.Popen([thirdparty_binary('gmm-acc-stats-twofeats'), model_path,
                                         feature_string, si_feature_string, "ark,s,cs:-", acc_path],
                                        stderr=log_file, stdin=ali_to_post_proc.stdout, env=os.environ)
            acc_proc.communicate()



def create_align_model(aligner: AlignerType) -> None:
    aligner.logger.info('Creating alignment model for speaker-independent features...')
    begin = time.time()
    log_directory = aligner.working_log_directory

    model_path = os.path.join(aligner.working_directory, 'final.mdl')
    align_model_path = os.path.join(aligner.working_directory, 'final.alimdl')
    arguments = [x.acc_stats_two_feats_arguments(aligner) for x in aligner.corpus.jobs]
    if aligner.use_mp:
        run_mp(acc_stats_two_feats_func, arguments, log_directory)
    else:
        run_non_mp(acc_stats_two_feats_func, arguments, log_directory)

    log_path = os.path.join(aligner.working_log_directory, 'align_model_est.log')
    with open(log_path, 'w', encoding='utf8') as log_file:

        acc_files = []
        for x in arguments:
            acc_files.extend(x.acc_paths.values())
        sum_proc = subprocess.Popen([thirdparty_binary('gmm-sum-accs'),
                                     '-'] + acc_files,
                                    stderr=log_file, stdout=subprocess.PIPE, env=os.environ)
        est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                     "--remove-low-count-gaussians=false",
                                     f'--power={aligner.power}',
                                     model_path,
                                     "-",
                                     align_model_path], stdin=sum_proc.stdout,
                                    stderr=log_file, env=os.environ)
        est_proc.communicate()
        if not aligner.debug:
            for f in acc_files:
                os.remove(f)

    aligner.logger.debug(f'Alignment model creation took {time.time() - begin}')


def lda_acc_stats_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ali_paths: Dict[str, str],
    model_path: str,
    lda_options: ConfigDict,
    acc_paths: Dict[str, str]) -> None:

    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            ali_path = ali_paths[dict_name]
            feature_string = feature_strings[dict_name]
            acc_path = acc_paths[dict_name]
            ali_to_post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                                 f'ark:{ali_path}',
                                                 'ark:-'],
                                                stderr=log_file, stdout=subprocess.PIPE, env=os.environ)
            weight_silence_post_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'),
                                                         f"{lda_options['boost_silence']}", lda_options['silence_csl'],
                                                         model_path,
                                                         'ark:-', 'ark:-'],
                                                        stdin=ali_to_post_proc.stdout,
                                                        stderr=log_file, stdout=subprocess.PIPE, env=os.environ)
            acc_lda_post_proc = subprocess.Popen([thirdparty_binary('acc-lda'),
                                                  f"--rand-prune={lda_options['random_prune']}",
                                                  model_path,
                                                  feature_string,
                                                  'ark,s,cs:-',
                                                  acc_path],
                                                 stdin=weight_silence_post_proc.stdout,
                                                 stderr=log_file, env=os.environ)
            acc_lda_post_proc.communicate()


def lda_acc_stats(aligner: LdaTrainer) -> None:
    """
    Multiprocessing function that accumulates LDA statistics

    See:

    - http://kaldi-asr.org/doc/ali-to-post_8cc.html
    - http://kaldi-asr.org/doc/weight-silence-post_8cc.html
    - http://kaldi-asr.org/doc/acc-lda_8cc.html
    - http://kaldi-asr.org/doc/est-lda_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_lda_mllt.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    directory : str
        Directory of LDA+MLLT training
    split_directory : str
        Directory of training data split into the number of jobs
    align_directory : str
        Directory of previous alignment
    config : :class:`~aligner.config.LdaMlltConfig`
        Configuration object for training
    ci_phones : str
        Colon-separated list of context-independent phones
    num_jobs : int
        The number of processes to use in calculation

    """
    arguments = [x.lda_acc_stats_arguments(aligner) for x in aligner.corpus.jobs]

    if aligner.use_mp:
        run_mp(lda_acc_stats_func, arguments, aligner.working_log_directory)
    else:
        run_non_mp(lda_acc_stats_func, arguments, aligner.working_log_directory)

    log_path = os.path.join(aligner.working_log_directory, 'lda_est.log')
    acc_list = []
    for x in arguments:
        acc_list.extend(x.acc_paths.values())
    with open(log_path, 'w', encoding='utf8') as log_file:
        est_lda_proc = subprocess.Popen([thirdparty_binary('est-lda'),
                                         f"--write-full-matrix={os.path.join(aligner.working_directory, 'full.mat')}",
                                         f'--dim={aligner.lda_dimension}',
                                         os.path.join(aligner.working_directory, 'lda.mat')] + acc_list,
                                        stderr=log_file, env=os.environ)
        est_lda_proc.communicate()


def calc_lda_mllt_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ali_paths: Dict[str, str],
    model_path: str,
    lda_options: ConfigDict,
    macc_paths: Dict[str, str]) -> None:

    # Estimating MLLT
    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            ali_path = ali_paths[dict_name]
            feature_string = feature_strings[dict_name]
            macc_path = macc_paths[dict_name]
            post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                          f"ark:{ali_path}", 'ark:-'],
                                         stdout=subprocess.PIPE, stderr=log_file, env=os.environ)

            weight_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'), '0.0',
                                            lda_options['silence_csl'], model_path, 'ark:-',
                                            'ark:-'],
                                           stdin=post_proc.stdout, stdout=subprocess.PIPE, stderr=log_file, env=os.environ)
            acc_proc = subprocess.Popen([thirdparty_binary('gmm-acc-mllt'),
                                         f"--rand-prune={lda_options['random_prune']}",
                                         model_path,
                                         feature_string,
                                         'ark,s,cs:-',
                                         macc_path],
                                        stdin=weight_proc.stdout, stderr=log_file, env=os.environ)
            acc_proc.communicate()


def calc_lda_mllt(aligner: LdaTrainer) -> None:
    """
    Multiprocessing function that calculates LDA+MLLT transformations

    See:

    - http://kaldi-asr.org/doc/ali-to-post_8cc.html
    - http://kaldi-asr.org/doc/weight-silence-post_8cc.html
    - http://kaldi-asr.org/doc/gmm-acc-mllt_8cc.html
    - http://kaldi-asr.org/doc/est-mllt_8cc.html
    - http://kaldi-asr.org/doc/gmm-transform-means_8cc.html
    - http://kaldi-asr.org/doc/compose-transforms_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_lda_mllt.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    directory : str
        Directory of LDA+MLLT training
    data_directory : str
        Directory of training data split into the number of jobs
    sil_phones : str
        Colon-separated list of silence phones
    num_jobs : int
        The number of processes to use in calculation
    config : :class:`~aligner.config.LdaMlltConfig`
        Configuration object for training
    initial : bool
        Flag for first iteration
    iteration : int
        Current iteration

    """
    jobs = [x.calc_lda_mllt_arguments(aligner) for x in aligner.corpus.jobs]

    if aligner.use_mp:
        run_mp(calc_lda_mllt_func, jobs, aligner.working_log_directory)
    else:
        run_non_mp(calc_lda_mllt_func, jobs, aligner.working_log_directory)


    log_path = os.path.join(aligner.working_log_directory, f'transform_means.{aligner.iteration}.log')
    previous_mat_path = os.path.join(aligner.working_directory, 'lda.mat')
    new_mat_path = os.path.join(aligner.working_directory, 'lda_new.mat')
    composed_path = os.path.join(aligner.working_directory, 'lda_composed.mat')
    with open(log_path, 'a', encoding='utf8') as log_file:
        macc_list = []
        for x in jobs:
            macc_list.extend(x.macc_paths.values())
        subprocess.call([thirdparty_binary('est-mllt'),
                         new_mat_path]
                        + macc_list,
                        stderr=log_file, env=os.environ)
        subprocess.call([thirdparty_binary('gmm-transform-means'),
                         new_mat_path,
                         aligner.current_model_path, aligner.current_model_path],
                        stderr=log_file, env=os.environ)

        if os.path.exists(previous_mat_path):
            subprocess.call([thirdparty_binary('compose-transforms'),
                             new_mat_path,
                             previous_mat_path,
                             composed_path],
                            stderr=log_file, env=os.environ)
            os.remove(previous_mat_path)
            os.rename(composed_path, previous_mat_path)
        else:
            os.rename(new_mat_path, previous_mat_path)



def map_acc_stats_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    model_path: str,
    ali_paths: Dict[str, str],
    acc_paths: Dict[str, str]) -> None:

    with open(log_path, 'w', encoding='utf8') as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            acc_path = acc_paths[dict_name]
            ali_path = ali_paths[dict_name]
            acc_proc = subprocess.Popen([thirdparty_binary('gmm-acc-stats-ali'), model_path,
                                         feature_string, f"ark,s,cs:{ali_path}", acc_path],
                                        stderr=log_file, env=os.environ)
            acc_proc.communicate()

def train_map(aligner: AdaptingAligner) -> None:
    """
    Source: https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_map.sh
    """
    begin = time.time()
    initial_mdl_path = os.path.join(aligner.working_directory, '0.mdl')
    final_mdl_path = os.path.join(aligner.working_directory, 'final.mdl')
    log_directory = aligner.working_log_directory
    os.makedirs(log_directory, exist_ok=True)

    jobs = [x.map_acc_stats_arguments(aligner) for x in aligner.corpus.jobs]
    if aligner.use_mp:
        run_mp(map_acc_stats_func, jobs, log_directory)
    else:
        run_non_mp(map_acc_stats_func, jobs, log_directory)
    log_path = os.path.join(aligner.working_log_directory, 'map_model_est.log')
    occs_path = os.path.join(aligner.working_directory, 'final.occs')
    with open(log_path, 'w', encoding='utf8') as log_file:
        acc_files = []
        for j in jobs:
            acc_files.extend(j.acc_paths.values())
        sum_proc = subprocess.Popen([thirdparty_binary('gmm-sum-accs'),
                                     '-'] + acc_files,
                                    stderr=log_file, stdout=subprocess.PIPE, env=os.environ)
        ismooth_proc = subprocess.Popen([thirdparty_binary('gmm-ismooth-stats'),
                                         '--smooth-from-model',
                                         f'--tau={aligner.mapping_tau}',
                                         initial_mdl_path,
                                         '-', '-'
                                         ], stderr=log_file, stdin=sum_proc.stdout, stdout=subprocess.PIPE, env=os.environ)
        est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                     '--update-flags=m',
                                     f'--write-occs={occs_path}',
                                     '--remove-low-count-gaussians=false',
                                     initial_mdl_path,
                                         '-',
                                     final_mdl_path,
                                     ], stdin=ismooth_proc.stdout, stderr=log_file, env=os.environ)
        est_proc.communicate()

    initial_alimdl_path = os.path.join(aligner.working_directory, '0.alimdl')
    final_alimdl_path = os.path.join(aligner.working_directory, '0.alimdl')
    if os.path.exists(initial_alimdl_path):
        aligner.speaker_independent = True
        jobs = [x.map_acc_stats_arguments(aligner) for x in aligner.corpus.jobs]
        if aligner.use_mp:
            run_mp(map_acc_stats_func, jobs, log_directory)
        else:
            run_non_mp(map_acc_stats_func, jobs, log_directory)

        log_path = os.path.join(aligner.working_log_directory, 'map_model_est.log')
        with open(log_path, 'w', encoding='utf8') as log_file:
            acc_files = []
        for j in jobs:
            acc_files.extend(j.acc_paths)
            sum_proc = subprocess.Popen([thirdparty_binary('gmm-sum-accs'),
                                         '-'] + acc_files,
                                        stderr=log_file, stdout=subprocess.PIPE, env=os.environ)
            ismooth_proc = subprocess.Popen([thirdparty_binary('gmm-ismooth-stats'),
                                             '--smooth-from-model',
                                             f'--tau={aligner.mapping_tau}',
                                             initial_alimdl_path,
                                             '-', '-'
                                             ], stderr=log_file, stdin=sum_proc.stdout, stdout=subprocess.PIPE, env=os.environ)
            est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                         '--update-flags=m',
                                         '--remove-low-count-gaussians=false',
                                         initial_alimdl_path,
                                         '-',
                                         final_alimdl_path,
                                         ], stdin=ismooth_proc.stdout, stderr=log_file, env=os.environ)
            est_proc.communicate()

    aligner.logger.debug(f'Mapping models took {time.time() - begin}')