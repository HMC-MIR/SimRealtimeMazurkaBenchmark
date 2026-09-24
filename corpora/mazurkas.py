"""The Chopin Mazurka corpus, from the Mazurka Project (http://mazurka.org.uk/)."""

from corpora.base import Corpus

# Recordings whose beat annotations cannot be matched beat for beat with the other
# recordings of their piece, because the performer plays a different form: Koczalski
# has 196 annotated beats where every other Op. 68 No. 3 recording has 184, and
# Ginzburg 327 where every other Op. 17 No. 4 recording has 399. Evaluation compares
# annotations beat by beat, so these two cannot be scored and are left out; this
# takes the Mazurka test set from 3852 to 3802 pairs. (Ginzburg's piece is only used
# for training, and the checked-in training lists already omit it.)
EXCLUDED_RECORDINGS = frozenset({
    "Chopin_Op017No4/Chopin_Op017No4_Ginzburg-1957_pid9156-10",
    "Chopin_Op068No3/Chopin_Op068No3_Koczalski-1948_pid9140-05",
})

MAZURKAS = Corpus(
    name="mazurkas",
    audio_root="Chopin_Mazurkas/wav_22050_mono",
    annot_root="Chopin_Mazurkas/annotations_beat",
    annot_ext=".beat",
    excluded=EXCLUDED_RECORDINGS,
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
