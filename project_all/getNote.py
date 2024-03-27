import numpy as np
import librosa
from crepe_core import predict


def decide_note(f0, KEY):
    degrees_in_key = librosa.key_to_degrees(KEY)
    degrees_in_key = np.append(degrees_in_key, degrees_in_key[0] + 12)

    midi_note = librosa.hz_to_midi(f0)
    degree = midi_note % 12
    closest_id = np.argmin(np.abs(degrees_in_key - degree))
    diff = degree - degrees_in_key[closest_id]

    midi_note -= diff
    return midi_note


def yin_getNote(indata, SAMPLE_RATE, KEY, MODEL_CAPACITY):
    # get the pitch
    pitches = librosa.yin(y=indata, 
                          fmin=65, fmax=3000, 
                          sr=SAMPLE_RATE, 
                          frame_length=2048, 
                          trough_threshold=0.1)
    f0 = np.median(pitches[0])

    # Convert pitch values to MIDI note numbers
    return decide_note(f0, KEY)


def pyin_getNote(indata, SAMPLE_RATE, KEY, MODEL_CAPACITY):
    # get the pitch
    f0, _, voiced_prob = librosa.pyin(y=indata,
                                        fmin=librosa.note_to_hz('C2'),
                                        fmax=librosa.note_to_hz('C6'),
                                        sr=SAMPLE_RATE,
                                        frame_length=2048)
    # check whether this frame is voiced
    f0 = f0[voiced_prob > 0.2]
    if len(f0) < 3:
        return -1
        
    # Convert pitch values to MIDI note numbers
    f0 = np.median(f0)
    return decide_note(f0, KEY)
  


def crepe_getNote(indata, SAMPLE_RATE, KEY, MODEL_CAPACITY):

    # Get the pitches using librosa
    frequency, confidence = predict(
        indata, SAMPLE_RATE,
        model_capacity=MODEL_CAPACITY,
        viterbi=False, 
        verbose=0)
    frequency = frequency[confidence > 0.3]
    # confidence = confidence[confidence > 0.3]
    if not np.any(frequency):
        return -1
    # pitch = frequency @ confidence / np.sum(confidence)
    f0 = np.median(frequency)

    # Convert pitch values to MIDI note numbers
    midi_note = round(librosa.core.hz_to_midi(f0))
    return midi_note
