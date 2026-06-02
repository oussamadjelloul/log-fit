"""HDFS line canonicalization — replicates the logfit-project preprocessing.

Pure utility, no methodology decisions. Reproduces the exact volatile-field
normalization the LogFiT authors applied in their published dataset
`logfit-project/hdfsv1-grouped-labeled` (Hugging Face). The recipe was
reverse-engineered from that dataset and validated BYTE-FOR-BYTE against it on
3,500 blocks (2,000 normal + 1,500 anomaly): 3500/3500 exact match.

Per-line canonical form is:
    "<level> <component> <normalized content>"
with the date/time/pid header dropped (the LogFiT text field carries no
timestamp; timestamps live in separate columns in their dataset).

Substitution ORDER is load-bearing — applied as written, or tokens get
captured by the wrong rule:
    1. IP:port   ( /10.0.0.1:50010 | 10.0.0.1:50010 ) -> IPADDRPORT
    2. bare IP   ( /10.0.0.1 | 10.0.0.1 )             -> IPADDR   (after #1)
    3. file path ( /any/slash/rooted/token )          -> FILEPATH (after IP rules)
    4. block id  ( blk_-123 | blk_123 )               -> blk_-NUM / blk_NUM (sign kept)
    5. integer   ( leftover digits )                  -> NUM      (LAST)
"""

from __future__ import annotations

import re

_IPPORT = re.compile(r"/?\d{1,3}(?:\.\d{1,3}){3}:\d+")   # IP:port, optional leading /
_IP = re.compile(r"/?\d{1,3}(?:\.\d{1,3}){3}")            # bare IP, optional leading /
_PATH = re.compile(r"/\S+")                               # slash-rooted file path
_BLK = re.compile(r"blk_(-?)\d+")                         # block id, sign preserved
_NUM = re.compile(r"\d+")                                 # any remaining integer


def canonicalize_content(content: str) -> str:
    """Normalize one log line's CONTENT field per the validated recipe.

    Order matters — see module docstring.
    """
    content = _IPPORT.sub("IPADDRPORT", content)
    content = _IP.sub("IPADDR", content)
    content = _PATH.sub("FILEPATH", content)
    content = _BLK.sub(r"blk_\1NUM", content)
    content = _NUM.sub("NUM", content)
    return content


def canonical_line(level: str, component: str, content: str) -> str:
    """Assemble one canonicalized sentence: '<level> <component> <content*>'.

    The date/time/pid header is intentionally absent (dropped to match the
    LogFiT text representation).
    """
    return f"{level} {component} {canonicalize_content(content)}"
