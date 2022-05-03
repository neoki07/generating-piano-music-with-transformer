from tensor2tensor.data_generators import text_encoder

SF2_PATH = "content/Yamaha-C5-Salamander-JNv5.1.sf2"
SAMPLE_RATE = 16000

# Decode a list of IDs.
def decode(ids, encoder):
    ids = list(ids)
    if text_encoder.EOS_ID in ids:
        ids = ids[: ids.index(text_encoder.EOS_ID)]
    return encoder.decode(ids)
