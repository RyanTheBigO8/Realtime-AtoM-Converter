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


'''Global Constants'''
DURATION = 3
SAMPLE_RATE = None
BLOCK_SIZE = None
STD_ENERGY = None
MIN_ENERGY = None
KEY = None
ALGORITHM = None
MODEL_CAPACITY = None
getNote_func = None   # function pointer to the getNote function

'''Global Variables'''
cur_note = -1                
pre_notes = [-1, -1, -1]     
pre_velo = [-1, -1, -1]
call_count = 0

def process_input():
    from getNote import yin_getNote, pyin_getNote, crepe_getNote
    global SAMPLE_RATE, BLOCK_SIZE, ALGORITHM, MODEL_CAPACITY, getNote_func
    
    # get the block size and getNote function
    if ALGORITHM == '0': # yin
        ALGORITHM = 'yin'
        BLOCK_SIZE = 2048
        getNote_func = yin_getNote
    elif ALGORITHM == '1': # pyin
        ALGORITHM = 'pyin'
        BLOCK_SIZE = 4096
        getNote_func = pyin_getNote
    elif ALGORITHM == '2': # crepe
        ALGORITHM = 'crepe'
        BLOCK_SIZE = 2048
        getNote_func = crepe_getNote
        SAMPLE_RATE = 16000.0

    # print the information
    print("[CONFIGURATION SUMMARY]: ")
    print(f"- device = {device_info['name']}")
    print(f"- sample rate = {SAMPLE_RATE} samples/sec")
    print(f"- blocksize = {BLOCK_SIZE} samples")
    print(f"- algorithm = {ALGORITHM}")
    print(f"- key = {KEY}")
    print(f"- standard energy = {STD_ENERGY}")

    return

def audio_callback(indata, frames, time, status):

    global cur_note, pre_notes, pre_velo
    note_on, note_off = [], []
    if status:
        print(status, flush=True)
    
    indata = indata.T

    # get the energy [CHECKPOINT 1]
    rms_energy = librosa.feature.rms(y=indata)[0]
    energy = np.mean(rms_energy)
    if energy < MIN_ENERGY:
        if cur_note != -1:
            print("Sending NoteOff event.")
            note_off = [NOTE_OFF, cur_note, 10]
            midiout.send_message(note_off)
            cur_note = -1
        return

    # convert energy to velocity
    cur_velo = round(75 * math.sqrt(energy / STD_ENERGY))
    cur_velo = min(127, cur_velo)

    # crop 'indata'
    if (ALGORITHM == 'pyin'):
        indata = indata[:2048]
    elif (ALGORITHM == 'crepe'):
        indata = indata.T[:1024]
    else:
        pass

    # Apply a low pass filter on the audio clip
    if ALGORITHM != 'crepe':
      b, a = signal.butter(2, 8000, btype='low', analog=False, fs=SAMPLE_RATE)
      indata = signal.lfilter(b, a, indata)

    # get the midi note [CHECKPOINT 2]
    temp = getNote_func(indata, SAMPLE_RATE, KEY, MODEL_CAPACITY)
    if (temp == -1):  # if no note detected
        if cur_note != -1:
            print("Sending NoteOff event.")
            note_off = [NOTE_OFF, cur_note, 10]
            midiout.send_message(note_off)
            cur_note = -1
        return

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
        print(f"Detected MIDI Note: {librosa.midi_to_note(cur_note)}", cur_velo, flush=True)
    
    # update 'pre_notes' and 'pre_velo'
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


def process_sample_recording(audio_data):
    global STD_ENERGY, MIN_ENERGY

    start_idx = len(audio_data) // 3
    end_idx = 2 * len(audio_data) // 3
    rms_energy = librosa.feature.rms(y=audio_data[start_idx:end_idx])[0]
    STD_ENERGY = np.mean(rms_energy)
    MIN_ENERGY = STD_ENERGY * 0.7


try:
    # Welcome message
    print("==============================================")
    print("|| Welcome to the Real-Time AtoM Converter! ||")
    print("==============================================")
    print()

    # select pitch estimation algorithm
    print("[STEP 1]: Select a Pitch Estimation Algorithm")
    print("Algorithms:")
    print("[0] yin")
    print("[1] pyin")
    print("[2] crepe")
    while True:
        ALGORITHM = input("Your choice (0, 1, or 2): ")
        if ALGORITHM in ['0', '1', '2']:
            break
        else:
            print("Error: Invalid input. Please enter 0, 1, or 2.")
    print()
    
    # if the user selects crepe, prompt for the model capacity
    if ALGORITHM == '2':
        print("The 'crepe' algorithm requires a model capacity to be specified.")
        print("Model Capacities:")
        print("(a) tiny")
        print("(b) small")
        print("(c) medium")
        print("(d) large")
        print("(e) full")
        while True:
            MODEL_CAPACITY = input("Your choice (tiny, small, medium, large, or full): ")
            if MODEL_CAPACITY in ['tiny', 'small', 'medium', 'large', 'full']:
                break
            else:
                print("Error: Invalid input. Please enter: tiny, small, medium, large, or full.")
        print()

    # prompt for key of song
    print("[STEP 2]: Specify the Key of Your Song")
    KEY = input("Your key: ")
    print()

    # get the device info and its default sample rate
    device_info = sd.query_devices(sd.default.device, 'input')
    if SAMPLE_RATE is None:
        SAMPLE_RATE = device_info['default_samplerate']

    # Record a sample voicefile
    print(f"[STEP 3]: Please sing any note for {DURATION} seconds.")
    input("Press ENTER to start recording...")
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to complete
    print("Recording complete.")
    process_sample_recording(audio_data.T[0])
    print()

    # process inputs and print info
    process_input()
    print()

    # Set up the plot 
    fig, ax = plt.subplots()
    timestamp, note, velocity = [], [], []
    ax.set_ylim(20, 100)
    start_time = time.time()

    ''' ---------- start threads ----------'''
    # THREAD 1: Midi Output Stream
    midiout, port_name = open_midioutput(None)
    # THREAD 2: Audio Stream
    stream = sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE, 
        callback=audio_callback)
    # THREAD 3: Plot Stream
    # animation = FuncAnimation(fig, update_plot, interval=10, blit=False)
    
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
    '''---------------------------------'''

except Exception as e:
    sys.exit()