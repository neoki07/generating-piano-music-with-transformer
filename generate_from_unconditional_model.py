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
parser.add_argument(
    "-p",
    "--primer",
    type=str,
    help="Midi file path for priming if not provided model will generate sample without priming.",
)
args = parser.parse_args()


class PianoPerformanceLanguageModelProblem(score2perf.Score2PerfProblem):
    @property
    def add_eos_symbol(self):
        return True


# Create input generator (so we can adjust priming and
# decode length on the fly).
def input_generator(targets, decode_length):
    while True:
        yield {
            "targets": np.array([targets], dtype=np.int32),
            "decode_length": np.array(decode_length, dtype=np.int32),
        }


def generate_ns_from_scratch(estimator, ckpt_path, unconditional_encoders):
    targets = []
    decode_length = 1024

    # Start the Estimator, loading from the specified checkpoint.
    input_fn = decoding.make_input_fn_from_generator(input_generator(targets, decode_length))
    unconditional_samples = estimator.predict(input_fn, checkpoint_path=ckpt_path)

    # Generate sample events.
    sample_ids = next(unconditional_samples)["outputs"]

    # Decode to NoteSequence.
    midi_filename = decode(sample_ids, encoder=unconditional_encoders["targets"])
    unconditional_ns = note_seq.midi_file_to_note_sequence(midi_filename)

    return unconditional_ns


def generate_ns_continuation(primer_ns, estimator, ckpt_path, unconditional_encoders):
    targets = unconditional_encoders["targets"].encode_note_sequence(primer_ns)

    # Remove the end token from the encoded primer.
    targets = targets[:-1]

    decode_length = max(0, 4096 - len(targets))
    if len(targets) >= 4096:
        print("Primer has more events than maximum sequence length; nothing will be generated.")

    # Start the Estimator, loading from the specified checkpoint.
    input_fn = decoding.make_input_fn_from_generator(input_generator(targets, decode_length))
    unconditional_samples = estimator.predict(input_fn, checkpoint_path=ckpt_path)

    # Generate sample events.
    sample_ids = next(unconditional_samples)["outputs"]

    # Decode to NoteSequence.
    midi_filename = decode(sample_ids, encoder=unconditional_encoders["targets"])
    ns = note_seq.midi_file_to_note_sequence(midi_filename)

    # Append continuation to primer.
    continuation_ns = note_seq.concatenate_sequences([primer_ns, ns])

    return continuation_ns


def main():
    if args.primer is not None and not os.path.isfile(args.primer):
        raise ValueError(f"'{args.primer}' is not a file path.")

    model_name = "transformer"
    hparams_set = "transformer_tpu"
    ckpt_path = "checkpoints/unconditional_model_16.ckpt"

    problem = PianoPerformanceLanguageModelProblem()
    unconditional_encoders = problem.get_feature_encoders()

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

    if args.primer is None:
        generated_ns = generate_ns_from_scratch(
            estimator,
            ckpt_path,
            unconditional_encoders,
        )
    else:
        # Use one of the provided primers.
        primer_ns = note_seq.midi_file_to_note_sequence(args.primer)

        # Handle sustain pedal in the primer.
        primer_ns = note_seq.apply_sustain_control_changes(primer_ns)

        # Trim to desired number of seconds.
        max_primer_seconds = 20
        if primer_ns.total_time > max_primer_seconds:
            print("Primer is longer than %d seconds, truncating." % max_primer_seconds)
            primer_ns = note_seq.extract_subsequence(primer_ns, 0, max_primer_seconds)

        # Remove drums from primer if present.
        if any(note.is_drum for note in primer_ns.notes):
            print("Primer contains drums; they will be removed.")
            notes = [note for note in primer_ns.notes if not note.is_drum]
            del primer_ns.notes[:]
            primer_ns.notes.extend(notes)

        # Set primer instrument and program.
        for note in primer_ns.notes:
            note.instrument = 1
            note.program = 0

        generated_ns = generate_ns_continuation(
            primer_ns,
            estimator,
            ckpt_path,
            unconditional_encoders,
        )

    # Write to output_dir.
    stem = f"unconditional_{time.strftime('%Y-%m-%d_%H%M%S')}"
    os.makedirs(args.output_dir, exist_ok=True)
    note_seq.note_sequence_to_midi_file(generated_ns, os.path.join(args.output_dir, f"{stem}.mid"))

    # Convert midi file to wave file.
    fs = FluidSynth(SF2_PATH, SAMPLE_RATE)
    fs.midi_to_audio(
        os.path.join(args.output_dir, f"{stem}.mid"),
        os.path.join(args.output_dir, f"{stem}.wav"),
    )


if __name__ == "__main__":
    main()
