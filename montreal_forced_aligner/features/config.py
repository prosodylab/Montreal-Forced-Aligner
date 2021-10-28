from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Dict, Any, Text, List
if TYPE_CHECKING:
    SpeakerCharacterType = Union[str, int]
    from ..corpus import CorpusType
    from ..corpus.classes import Job

import os
from ..exceptions import ConfigError
from ..config import BaseConfig


def make_safe(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


class FeatureConfig(BaseConfig):
    """
    Class to store configuration information about MFCC generation

    Parameters
    ----------
    directory : str
        Path to directory to save configuration files for Kaldi

    Attributes
    ----------
    directory : str
        Path of the directory to store outputs
    type : str
        Feature type, defaults to "mfcc"
    deltas : bool
        Flag for whether deltas from previous frames are included in the features, defaults to True
    lda : bool
        Flag for whether LDA is run on the features, requires an lda.mat to generate, defaults to False
    fmllr : bool
        Flag for whether speaker adaptation should be run, defaults to False
    use_energy : bool
        Flag for whether first coefficient should be used, defaults to False
    frame_shift : int
        number of milliseconds between frames, defaults to 10
    pitch : bool
        Flag for including pitch in features, currently nonfunctional, defaults to False
    splice_left_context : int or None
        Number of frames to splice on the left for calculating LDA
    splice_right_context : int or None
        Number of frames to splice on the right for calculating LDA
    use_mp : bool
        Flag for using multiprocessing, defaults to True
    """
    deprecated_flags = {'lda', 'deltas'}
    def __init__(self):
        self.type = 'mfcc'
        self.deltas = True
        self.fmllr = False
        self.lda = False
        self.use_energy = False
        self.frame_shift = 10
        self.snip_edges = True
        self.pitch = False
        self.low_frequency = 20
        self.high_frequency = 7800
        self.sample_frequency = 16000
        self.allow_downsample = True
        self.allow_upsample = True
        self.splice_left_context = 3
        self.splice_right_context = 3
        self.use_mp = True

    def params(self) -> Dict[Text, Any]:
        return {'type': self.type,
                'use_energy': self.use_energy,
                'frame_shift': self.frame_shift,
                'snip_edges': self.snip_edges,
                'low_frequency': self.low_frequency,
                'high_frequency': self.high_frequency,
                'sample_frequency': self.sample_frequency,
                'allow_downsample': self.allow_downsample,
                'allow_upsample': self.allow_upsample,
                'pitch': self.pitch,
                'fmllr': self.fmllr,
                'splice_left_context': self.splice_left_context,
                'splice_right_context': self.splice_right_context,
                }

    def mfcc_options(self) -> Dict[Text, Any]:
        """Return dictionary of parameters to use in computing MFCC features."""
        return {'use-energy': self.use_energy, 'frame-shift': self.frame_shift, 'low-freq': self.low_frequency,
                'high-freq': self.high_frequency, 'sample-frequency': self.sample_frequency,
                'allow-downsample': self.allow_downsample, 'allow-upsample': self.allow_upsample,
                'snip-edges': self.snip_edges}



    def update(self, data: Dict[Text, Any]) -> None:
        for k, v in data.items():
            if k in self.deprecated_flags:
                continue
            if not hasattr(self, k):
                raise ConfigError('No field found for key {}'.format(k))
            setattr(self, k, v)

    @property
    def feature_id(self) -> Text:
        return 'feats'


