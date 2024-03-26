import argparse
import queue
import sys
import time
import math


from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import librosa
import scipy.signal as signal
from rtmidi.midiutil import open_midioutput
from rtmidi.midiconstants import NOTE_OFF, NOTE_ON
import noisereduce as nr


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def parse_input():
    
    parser = argparse.ArgumentParser(
        sys.argv[0],
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
        help='input channels to plot (default: the first)')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-i', '--interval', type=float, default=10,
        help='minimum time between plot updates (default: %(default)s ms)')
    parser.add_argument(
        '-b', '--blocksize', type=int, default=2048, help='block size (in samples)')
    parser.add_argument(
        '-r', '--samplerate', type=float, help='sampling rate of audio device')
    
    
    args = parser.parse_args()

    device_info = sd.query_devices(args.device, 'input')
    if args.samplerate is None:
        args.samplerate = device_info['default_samplerate']

    print(f"device = {device_info['name']}")
    print(f"channels = {args.channels}")
    print(f"sample rate = {args.samplerate} samples/sec")
    print(f"blocksize = {args.blocksize} samples")
    print(f"update interval = {args.interval}ms")

    return args, parser


'''Global Variables'''
sample_rate = None
block_size = None
standard_energy = None
minimum_energy = None
cur_note = -1                
pre_notes = [-1, -1, -1]     
pre_velo = [-1, -1, -1]
KEY = None

def decide_note(f0):
    degrees_in_key = librosa.key_to_degrees(KEY)
    degrees_in_key = np.append(degrees_in_key, degrees_in_key[0] + 12)

    midi_note = librosa.hz_to_midi(f0)
    degree = midi_note % 12
    closest_id = np.argmin(np.abs(degrees_in_key - degree))
    diff = degree - degrees_in_key[closest_id]

    midi_note -= diff
    return midi_note

def audio_callback(indata, frames, time, status):

    global cur_note, pre_notes, pre_velo
    note_on, note_off = [], []
    if status:
        print(status, flush=True)
    
    indata = indata.T[:2048]

    # Apply a low pass filter on the audio clip
    b, a = signal.butter(2, 8000, btype='low', analog=False, fs=sample_rate)
    filtered_audio = signal.lfilter(b, a, indata)

    # get the pitch
    f0, _, voiced_prob = librosa.pyin(y=filtered_audio,
                                        fmin=librosa.note_to_hz('C2'),
                                        fmax=librosa.note_to_hz('C6'),
                                        sr=sample_rate,
                                        frame_length=2048)
    # check whether this frame is voiced
    f0 = f0[voiced_prob > 0.2]
    if len(f0) < 3:
        if cur_note != -1:
            print("Sending NoteOff event.")
            note_off = [NOTE_OFF, cur_note, 10]
            midiout.send_message(note_off)
            cur_note = -1
        return

    # get the energy and convert to velocity
    rms_energy = librosa.feature.rms(y=indata)[0]
    energy = np.mean(rms_energy)
    cur_velo = round(75 * math.sqrt(energy / standard_energy))
    cur_velo = min(127, cur_velo)

    # Convert pitch values to MIDI note numbers
    f0 = np.median(f0)
    temp = decide_note(f0)
    
    # update 'cur_note'
    if temp == pre_notes[-1] and temp == pre_notes[-2]:
        if temp != cur_note:
            if cur_note != -1:
                print("Sending NoteOff event.")
                note_off = [NOTE_OFF, cur_note, 10]
                midiout.send_message(note_off)
            print(f"Sending NoteOn event.")
            note_on = [NOTE_ON, temp, cur_velo]
            midiout.send_message(note_on)
        else:
            if cur_velo > (pre_velo[-1] + 5):
                print("Sending NoteOff event.")
                note_off = [NOTE_OFF, cur_note, pre_velo[-1]]
                midiout.send_message(note_off)
                print(f"Sending NoteOn event.")
                note_on = [NOTE_ON, temp, cur_velo]
                midiout.send_message(note_on)
        cur_note = temp
        pre_velo = pre_velo[1:] + [cur_velo]
        print(f"Detected MIDI Note: {librosa.midi_to_note(cur_note)}", cur_velo, energy, flush=True)
    
    # update 'pre_notes' and 'pre_velo'
    pre_notes = pre_notes[1:] + [temp]


def process_sample_recording(audio_data):
    global standard_energy, minimum_energy
    start_idx = len(audio_data) // 3
    end_idx = 2 * len(audio_data) // 3
    rms_energy = librosa.feature.rms(y=audio_data[start_idx:end_idx])[0]
    standard_energy = np.mean(rms_energy)
    print(f"Standard Energy: {standard_energy}")
    minimum_energy = standard_energy * 0.5

try:
    # parse inputs
    args, parser = parse_input()
    sample_rate = args.samplerate
    block_size = args.blocksize

    # select midi output port
    port = sys.argv[1] if len(sys.argv) > 1 else None

    # prompt for key of song
    KEY = input("Please specify the key of your song: ")

    # Record a sample voicefile
    duration = 3
    print(f"Please sing any note for {duration} seconds.")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=max(args.channels), dtype='float32')
    sd.wait()  # Wait for the recording to complete
    print("Recording complete.")
    process_sample_recording(audio_data.T[0])


    ''' start threads '''
    # THREAD 1: Audio Stream
    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=sample_rate, blocksize=block_size, callback=audio_callback)
    # THREAD 2: Midi Output Stream
    midiout, port_name = open_midioutput(port)
   
    try:
        with stream:
            while True:
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("**KEYBOARD INTERRUPT DETECTED**")
        stream.stop()
        stream.close()
        del midiout
        sys.exit(0)

except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
    sys.exit()