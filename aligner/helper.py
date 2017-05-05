import os
import sys
from collections import defaultdict


def thirdparty_binary(binary_name):
    return binary_name


def make_path_safe(path):
    return '"{}"'.format(path)


def load_text(path):
    with open(path, 'r', encoding='utf8') as f:
        text = f.read().strip().lower()
    return text


def make_safe(element):
    if isinstance(element, list):
        return ' '.join(map(make_safe, element))
    return str(element)

def thirdparty_path(binary_name):
    return os.path.join(os.path.split(os.path.split(__file__)[0])[0], 'thirdparty','bin', binary_name)