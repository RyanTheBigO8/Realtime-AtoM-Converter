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
minimum_energy = 0.0045
cur_note = -1                
pre_notes = [-1, -1, -1]     
pre_velo = [-1, -1, -1]
call_count = 0
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

    global cur_note, pre_notes, pre_velo, call_count
    note_on, note_off = [], []
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, flush=True)

    # get the energy
    rms_energy = librosa.feature.rms(y=indata.T)[0]
    energy = np.mean(rms_energy)
    if energy < minimum_energy:
        if cur_note != -1:
            print("Sending NoteOff event.")
            note_off = [NOTE_OFF, cur_note, 10]
            midiout.send_message(note_off)
            cur_note = -1
            call_count = 0
        return
    
    # convert energy to velocity
    '''
    cur_velo = round((np.mean(rms_energy) - 0.002) / 0.012 * 127)
    cur_velo = min(127, cur_velo)
    '''
    cur_velo = round(75 * energy / standard_energy)
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
    
    f0 = np.median(pitches[0])

    # Convert pitch values to MIDI note numbers
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
            call_count = 0
        else:
            if cur_velo > (pre_velo[-1] + 20) and call_count > 1:
                print("Sending NoteOff event.")
                note_off = [NOTE_OFF, cur_note, pre_velo[-1]]
                midiout.send_message(note_off)
                print(f"Sending NoteOn event.")
                note_on = [NOTE_ON, temp, cur_velo]
                midiout.send_message(note_on)
                call_count = 0
        call_count += 1
        cur_note = temp
        print(f"Detected MIDI Note: {librosa.midi_to_note(cur_note)}", cur_velo, flush=True)
    
    # update 'pre_notes' and 'pre_velo'
    pre_notes = pre_notes[1:] + [temp]
    pre_velo = pre_velo[1:] + [cur_velo]
    

def update_plot(frame):
    global cur_note
    
    current_time = time.time() - start_time
    timestamp.append(current_time)
    note.append(cur_note)

    # Update the x-axis limits dynamically based on the current time
    ax.set_xlim(max(0, current_time - 10), current_time)

    # Process the current midi_notes and update the plot
    plt.plot(timestamp, note, marker='o', markersize=2, linestyle='None', color='blue')


def process_sample_recording(audio_data):
    global standard_energy
    start_idx = len(audio_data) // 3
    end_idx = 2 * len(audio_data) // 3
    rms_energy = librosa.feature.rms(y=audio_data[start_idx:end_idx])[0]
    standard_energy = np.mean(rms_energy)
    print(standard_energy)


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

    # Set up the plot 
    fig, ax = plt.subplots()
    timestamp, note, velocity = [], [], []
    ax.set_ylim(20, 100)
    start_time = time.time()

    ''' start threads '''
    # THREAD 1: Audio Stream
    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=sample_rate, blocksize=block_size, callback=audio_callback)
    # THREAD 2: Midi Output Stream
    midiout, port_name = open_midioutput(port)
    # THREAD 3: Plot Stream
    # animation = FuncAnimation(fig, update_plot, interval=args.interval, blit=False)
    
    try:
        with stream:
            while True:
                # plt.show()
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
