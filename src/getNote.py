import numpy as np
import librosa
import crepe_core


def yin_getNote(indata, SAMPLE_RATE, KEY, MODEL_CAPACITY):
    # get the pitch
    pitches = librosa.yin(y=indata, 
                          fmin=65, fmax=3000, 
                          sr=SAMPLE_RATE, 
                          frame_length=2048, 
                          trough_threshold=0.1)
    f0 = np.median(pitches[0])

    # Convert pitch values to MIDI note numbers
    midi_note = round(librosa.core.hz_to_midi(f0))
    return midi_note


def pyin_getNote(indata, SAMPLE_RATE, KEY, MODEL_CAPACITY):
    # get the pitch
    f0, _, voiced_prob = librosa.pyin(y=indata,
                                        fmin=librosa.note_to_hz('B1'),
                                        fmax=librosa.note_to_hz('D6'),
                                        sr=SAMPLE_RATE,
                                        frame_length=2048)
    # check whether this frame is voiced
    f0 = f0[voiced_prob > 0.3]
    f0 = f0[~np.isnan(f0)]
    if len(f0) < 3:
        return -1
    
    # Convert pitch values to MIDI note numbers
    f0 = np.median(f0)
    midi_note = round(librosa.core.hz_to_midi(f0))
    return midi_note
  


def crepe_getNote(indata, SAMPLE_RATE, KEY, MODEL_CAPACITY):

    # Get the pitches using librosa
    frequency, confidence = crepe_core.predict(
        indata, SAMPLE_RATE,
        model_capacity=MODEL_CAPACITY,
        step_size=20,
        viterbi=False, 
        verbose=0)
    frequency = frequency[confidence > 0.5]
    # confidence = confidence[confidence > 0.3]
    if not np.any(frequency):
        return -1
    # pitch = frequency @ confidence / np.sum(confidence)
    f0 = np.median(frequency)

    # Convert pitch values to MIDI note numbers
    midi_note = round(librosa.core.hz_to_midi(f0))
    return midi_note
