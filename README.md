# Generating Piano Music with Transformer

This is a ported script from the original Music Transformer [notebook](https://colab.research.google.com/notebooks/magenta/piano_transformer/piano_transformer.ipynb).
Also included is a Dockerfile that makes it easy build the environment to run the script.

This project is inspired by [this repository](https://github.com/Elvenson/piano_transformer), but differs in the following:

* Follows newer versions of notebook (e.g. library versions used)
* Easily create an execution environment with docker and poetry
* Export as wave file in addition to MIDI file
* Only the original notebook generation method is implemented

## Getting Started


> You must have Docker installed on your computer.

1. Clone the repository

```sh
git clone https://github.com/ot07/generating-piano-music-with-transformer.git
cd generating-piano-music-with-transformer
```

2. Build a docker image using the `Dockerfile`

```sh
docker build -t generating-piano-music-with-transformer .
```

3. Run the script using the docker image created in the previous step

### Generate from Scratch

You can generate a piano performance from scratch using the unconditional model.

```sh
docker run -it --rm -v ${PWD}/output:output generating-piano-music-with-transformer \
  python generate_from_unconditional_model.py --output_dir=output
```

### Generate Continuation

By specifying a primer MIDI file in `--primer` argument,
you can generate a piano performance that is a continuation of the chosen primer.

```sh
docker run -it --rm -v ${PWD}/output:output generating-piano-music-with-transformer \
  python generate_from_unconditional_model.py --output_dir=output --primer=<primer midi file>
```

### Generate Accompaniment for Melody

By specifying a melody MIDI file in `--melody` argument,
You can generate a piano performance consisting of the chosen melody plus accompaniment.

```sh
docker run -it --rm -v ${PWD}/output:output generating-piano-music-with-transformer \
  python generate_from_melody_conditioned_model.py --output_dir=output --melody=<melody midi file>
```

## Inspiration

This is inspired by the following repository:

* [Music Transformer Script: A ported script from Google Music Transformer notebook](https://github.com/Elvenson/piano_transformer)
