import argparse
import os
import time

import note_seq
import numpy as np
from magenta.models.score2perf import score2perf
from midi2audio import FluidSynth
from tensor2tensor.utils import decoding, trainer_lib

from commons import SAMPLE_RATE, SF2_PATH, decode

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_dir", type=str, required=True, help="Midi output directory.")
parser.add_argument("-m", "--melody", type=str, required=True, help="Midi file path for melody")
args = parser.parse_args()


class MelodyToPianoPerformanceProblem(score2perf.AbsoluteMelody2PerfProblem):
    @property
    def add_eos_symbol(self):
        return True


# Create input generator.
def input_generator(inputs, decode_length):
    while True:
        yield {
            "inputs": np.array([[inputs]], dtype=np.int32),
            "targets": np.zeros([1, 0], dtype=np.int32),
            "decode_length": np.array(decode_length, dtype=np.int32),
        }


def generate_accompaniment_ns_for_melody(inputs, estimator, ckpt_path, melody_conditioned_encoders):
    # Generate sample events.
    decode_length = 4096

    # Start the Estimator, loading from the specified checkpoint.
    input_fn = decoding.make_input_fn_from_generator(input_generator(inputs, decode_length))
    melody_conditioned_samples = estimator.predict(input_fn, checkpoint_path=ckpt_path)

    sample_ids = next(melody_conditioned_samples)["outputs"]

    # Decode to NoteSequence.
    midi_filename = decode(sample_ids, encoder=melody_conditioned_encoders["targets"])
    accompaniment_ns = note_seq.midi_file_to_note_sequence(midi_filename)

    return accompaniment_ns


def main():
    if args.melody is not None and not os.path.isfile(args.melody):
        raise ValueError(f"'{args.melody}' is not a file path.")

    model_name = "transformer"
    hparams_set = "transformer_tpu"
    ckpt_path = "checkpoints/melody_conditioned_model_16.ckpt"

    problem = MelodyToPianoPerformanceProblem()
    melody_conditioned_encoders = problem.get_feature_encoders()

    # Set up HParams.
    hparams = trainer_lib.create_hparams(hparams_set=hparams_set)
    trainer_lib.add_problem_hparams(hparams, problem)
    hparams.num_hidden_layers = 16
    hparams.sampling_method = "random"

    # Set up decoding HParams.
    decode_hparams = decoding.decode_hparams()
    decode_hparams.alpha = 0.0
    decode_hparams.beam_size = 1

    # Create Estimator.
    run_config = trainer_lib.create_run_config(hparams)
    estimator = trainer_lib.create_estimator(model_name, hparams, run_config, decode_hparams=decode_hparams)

    # Extract melody from MIDI file.
    with open(args.melody, "rb") as f:
        melody_bytes = f.read()

    melody_ns = note_seq.midi_to_note_sequence(melody_bytes)
    melody_instrument = note_seq.infer_melody_for_sequence(melody_ns)
    notes = [note for note in melody_ns.notes if note.instrument == melody_instrument]
    del melody_ns.notes[:]
    melody_ns.notes.extend(sorted(notes, key=lambda note: note.start_time))
    for i in range(len(melody_ns.notes) - 1):
        melody_ns.notes[i].end_time = melody_ns.notes[i + 1].start_time
    inputs = melody_conditioned_encoders["inputs"].encode_note_sequence(melody_ns)

    accompaniment_ns = generate_accompaniment_ns_for_melody(inputs, estimator, ckpt_path, melody_conditioned_encoders)

    # Write to output_dir.
    stem = f"accompaniment_{time.strftime('%Y-%m-%d_%H%M%S')}"
    os.makedirs(args.output_dir, exist_ok=True)
    note_seq.note_sequence_to_midi_file(accompaniment_ns, os.path.join(args.output_dir, f"{stem}.mid"))

    # Convert midi file to wave file.
    fs = FluidSynth(SF2_PATH, SAMPLE_RATE)
    fs.midi_to_audio(
        os.path.join(args.output_dir, f"{stem}.mid"),
        os.path.join(args.output_dir, f"{stem}.wav"),
    )


if __name__ == "__main__":
    main()
