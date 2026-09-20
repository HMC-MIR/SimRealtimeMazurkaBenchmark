"""The Chopin Mazurka corpus, from the Mazurka Project (http://mazurka.org.uk/)."""

from corpora.base import Corpus

MAZURKAS = Corpus(
    name="mazurkas",
    audio_root="Chopin_Mazurkas/wav_22050_mono",
    annot_root="Chopin_Mazurkas/annotations_beat",
    annot_ext=".beat",
)

# The piece the train benchmarks are drawn from. Their cfg/*.pkl lists hold bare
# recording names, so the benchmark config pairs this corpus with
# default_piece_root=TRAIN_PIECE_ROOT.
TRAIN_PIECE_ROOT = "Chopin_Op017No4"

TEST_PIECE_ROOTS = [
    "Chopin_Op024No2",
    "Chopin_Op030No2",
    "Chopin_Op068No3",
]
