"""
The Vienna 4x22 Piano Corpus: 22 pianists playing each of 4 excerpts.

Collected by Werner Goebl on a Boesendorfer SE290 computer-monitored grand, so
every performance was captured as audio and as MIDI at once. Score-to-performance
alignments are distributed as match files at https://github.com/CPJKU/vienna4x22
(CC BY 4.0); the audio is a separate download (see scripts/vienna4x22/prepare.py).

Unlike the Mazurka corpus, all 22 performances of a piece share one instrument,
room and microphone setup. That isolates tempo and expressive variation from the
recording-condition variation that dominates the Mazurka set, so the two
benchmarks probe different things and their error rates are not interchangeable.

Beat annotations are not distributed with the corpus. They are derived from the
match files by scripts/vienna4x22/prepare.py, which writes them in the same
.beat format the Mazurka annotations use.
"""

from corpora.base import Corpus

VIENNA4X22 = Corpus(
    name="vienna4x22",
    audio_root="Vienna4x22/wav_22050_mono",
    annot_root="Vienna4x22/annotations_beat",
    annot_ext=".beat",
)

PIECE_ROOTS = [
    "Chopin_op10_no3",
    "Chopin_op38",
    "Mozart_K331_1st-mov",
    "Schubert_D783_no15",
]
