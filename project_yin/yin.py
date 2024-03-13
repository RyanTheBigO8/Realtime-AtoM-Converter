import argparse
import queue
import sys
import time
import logging

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import librosa
import scipy.signal as signal
from rtmidi.midiutil import open_midioutput
from rtmidi.midiconstants import NOTE_OFF, NOTE_ON

print("hello")

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

args, parser = parse_input()

sample_rate = args.samplerate
block_size = args.blocksize
energy_threshold = 0.004
cur_note = -1                    # note number
pre_notes = [-1, -1, -1, -1]     # note numbers
replace_count = 0


def audio_callback(indata, frames, time, status):

    global cur_note, pre_notes
    note_on, note_off = [], []
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, flush=True)

    # get the energy
    rms_energy = librosa.feature.rms(y=indata.T)[0]
    if np.mean(rms_energy) < energy_threshold:
        if cur_note != -1:
            print("Sending NoteOff event.")
            note_off = [NOTE_OFF, cur_note, 10]
            midiout.send_message(note_off)
            cur_note = -1
        return
    
    # convert energy to velocity
    cur_velo = round((np.mean(rms_energy) - 0.002) / 0.015 * 127)
    cur_velo = min(127, cur_velo)

    # Apply a low pass filter on the audio clip
    b, a = signal.butter(2, 8000, btype='low', analog=False, fs=sample_rate)
    filtered_audio = signal.lfilter(b, a, indata.T)
    
    # Get the pitches using librosa
    pitches = librosa.yin(y=filtered_audio, 
                          fmin=65, fmax=3000, 
                          sr=sample_rate, 
                          frame_length=2048, 
                          trough_threshold=0.2)
    median_pitch = np.median(pitches[0])

    # Convert pitch values to MIDI note numbers
    temp = round(librosa.core.hz_to_midi(median_pitch))
    
    # update 'cur_note'
    if temp == pre_notes[-1] and temp == pre_notes[-2]:
        if temp != cur_note and temp != -1:
            if cur_note != -1:
                print("Sending NoteOff event.")
                note_off = [NOTE_OFF, cur_note, 10]
                midiout.send_message(note_off)
            print(f"Sending NoteOn event. Velo = {cur_velo}")
            note_on = [NOTE_ON, temp, cur_velo]
            midiout.send_message(note_on)
        cur_note = temp
        print(f"Detected MIDI Note: {librosa.midi_to_note(cur_note)}", cur_velo, flush=True)
    
    # update 'pre_notes'
    pre_notes = pre_notes[1:] + [temp]
    

def update_plot(frame):
    global cur_note
    
    current_time = time.time() - start_time
    timestamp.append(current_time)
    note.append(cur_note)

    # Update the x-axis limits dynamically based on the current time
    ax.set_xlim(max(0, current_time - 10), current_time)

    # Process the current midi_notes and update the plot
    plt.plot(timestamp, note, marker='o', markersize=2, linestyle='None', color='blue')
    

try:
    # Set up the plot
    fig, ax = plt.subplots()
    timestamp, note, velocity = [], [], []
    ax.set_ylim(20, 100)
    start_time = time.time()

    # select midi output port
    port = sys.argv[1] if len(sys.argv) > 1 else None

    ### start threads ###
    # THREAD 1: Audio Stream
    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, blocksize=block_size, callback=audio_callback)
    # THREAD 2: Midi Output Stream
    midiout, port_name = open_midioutput(port)
    # THREAD 3: Plot Stream
    # animation = FuncAnimation(fig, update_plot, interval=args.interval, blit=False)
    
    with stream:
        # plt.show()
        sd.sleep(1000000)

    del midiout
    print('Exit')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
    sys.exit()