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
from ANSI_colors import Color


'''Global Constants'''
DURATION = 3
SAMPLE_RATE = None
BLOCK_SIZE = None
STD_ENERGY = None
MIN_ENERGY = None
MIN_VELO = 20
KEY = None
ALGORITHM = None
MODEL_CAPACITY = None
getNote_func = None   # function pointer to the getNote function
tuner = True

'''Global Variables'''
cur_note = -1                
pre_notes = [-1, -1, -1]     
pre_velo = [0, 0, 0]
note_count = 0
NoteOn_velo = 0
NoteOff_velo = 0
decay_amount = 10     # 'decay_amount' = 'NoteOn_velo' - 'NoteOff_velo'
NoteOn_seq = np.array([], dtype=str)


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
        BLOCK_SIZE = 1024
        getNote_func = crepe_getNote
        SAMPLE_RATE = 16000.0

    # print the information
    print(Color.YELLOW + "[CONFIGURATION SUMMARY]:" + Color.END)
    print()
    print(Color.PURPLE + "- device" + Color.END + f" = {device_info['name']}")
    print(Color.PURPLE + "- sample rate" + Color.END + f" = {SAMPLE_RATE} samples/sec")
    print(Color.PURPLE + "- blocksize" + Color.END + f" = {BLOCK_SIZE} samples")
    print(Color.PURPLE + "- algorithm" + Color.END + f" = {ALGORITHM}")
    print(Color.PURPLE + "- key" + Color.END + f" = {KEY}")
    print(Color.PURPLE + "- standard energy" + Color.END + f" = {STD_ENERGY}")

    return


def tune_note(midi_note, KEY):
    degrees_in_key = librosa.key_to_degrees(KEY)
    degrees_in_key = np.append(degrees_in_key, degrees_in_key[0] + 12)
    # degrees_in_key = np.append(degrees_in_key, 1)

    degree = midi_note % 12
    closest_id = np.argmin(np.abs(degrees_in_key - degree))
    diff = degree - degrees_in_key[closest_id]

    midi_note -= diff
    return midi_note


def Send_NoteOn(note, velocity):
    global cur_note, NoteOn_velo, NoteOn_seq

    print("Sending NoteOn event.")
    midiout.send_message([NOTE_ON, note, velocity])
    NoteOn_seq = np.append(NoteOn_seq, str(note))
    NoteOn_velo = velocity
    cur_note = note


def Send_NoteOff(note, velocity):
    global cur_note, note_count, NoteOff_velo, decay_amount

    print("Sending NoteOff event.")
    midiout.send_message([NOTE_OFF, note, velocity])
    note_count += 1
    NoteOff_velo = velocity
    diff = NoteOn_velo - NoteOff_velo
    if diff > 10 and diff < 75 and note_count > 1:
        decay_amount = 0.6 * decay_amount + 0.4 * diff
    cur_note = -1
    print(decay_amount, flush=True)


def audio_callback(indata, frames, time, status):
    global cur_note, pre_notes, pre_velo, note_count 
    global NoteOn_velo, NoteOff_velo, decay_amount, NoteOn_seq

    if status:
        # print(status, flush=True)
        return
    
    indata = indata.T

    # get the energy
    rms_energy = librosa.feature.rms(y=indata)[0]
    energy = np.mean(rms_energy)

    # convert energy to velocity
    cur_velo = round(75 * math.sqrt(energy / STD_ENERGY))
    cur_velo = min(127, cur_velo)

    # determine voiced or not [CHECKPOINT 1]
    if energy < MIN_ENERGY:
    # if cur_velo < MIN_VELO:
        if cur_note != -1 and pre_notes[-1] == -1:
            Send_NoteOff(cur_note, cur_velo)
        pre_notes = pre_notes[1:] + [-1]
        return

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
        if cur_note != -1 and pre_notes[-1] == -1:
            Send_NoteOff(cur_note, cur_velo)
        pre_notes = pre_notes[1:] + [-1]
        return
    
    # Apply auto-tuner if enabled
    if tuner:
        temp = tune_note(temp, KEY)

    # update 'cur_note'
    if temp == pre_notes[-1]:
        if temp != cur_note:
            if cur_note != -1:
                Send_NoteOff(cur_note, pre_velo[-2])
            Send_NoteOn(temp, cur_velo)
        else:
            diff = cur_velo - pre_velo[-1]
            # if (cur_velo - pre_velo[-1] > 0.8 * decay_amount):
            if diff > 8:
                Send_NoteOff(cur_note, pre_velo[-1])
                Send_NoteOn(temp, cur_velo)

        pre_velo = pre_velo[1:] + [cur_velo]
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


def process_sample_recording(audio_data):
    global STD_ENERGY, MIN_ENERGY

    start_idx = len(audio_data) // 3
    end_idx = 2 * len(audio_data) // 3
    rms_energy = librosa.feature.rms(y=audio_data[start_idx:end_idx])[0]
    STD_ENERGY = np.mean(rms_energy)
    MIN_ENERGY = STD_ENERGY * 0.5


try:
    # Welcome message
    print(Color.PURPLE + "==============================================" + Color.END)
    print(Color.PURPLE + "||" + Color.END + Color.BOLD + " Welcome to the Real-Time AtoM Converter! " + Color.END + Color.PURPLE + "||" + Color.END)
    print(Color.PURPLE + "==============================================" + Color.END)
    print()

    # get the device info and its default sample rate
    device_info = sd.query_devices(sd.default.device, 'input')
    print(Color.PURPLE + "Input Device: " + Color.END + Color.UNDERLINE + f" {device_info['name']} " + Color.END)
    print()
    if SAMPLE_RATE is None:
        SAMPLE_RATE = device_info['default_samplerate']

    # select pitch estimation algorithm
    print(Color.YELLOW + "[STEP 1]: " + Color.END + "Select a Pitch Estimation Algorithm")
    print()
    print("[0] yin")
    print("[1] pyin")
    print("[2] crepe")
    while True:
        ALGORITHM = input("Your choice (0, 1, or 2): ")
        if ALGORITHM in ['0', '1', '2']:
            break
        else:
            print(Color.RED + "Error: Invalid input. Please enter 0, 1, or 2." + Color.END)
    print()
    
    # if the user selects crepe, prompt for the model capacity
    if ALGORITHM == '2':
        print(Color.YELLOW + "The 'crepe' algorithm requires a model capacity to be specified." + Color.END)
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
                print(Color.RED + "Error: Invalid input. Please enter: tiny, small, medium, large, or full." + Color.END)
        print()

    # Auto-tuner settings
    print(Color.YELLOW + "[STEP 2]: " + Color.END + "Auto-Tuner Settings")
    print()
    enable = input("Enable Auto-Tuner? (y/n): ")
    print()
    if enable == 'n':
        tuner = False
    else:
        tuner = True
        print("Please specify the key of your song")
        KEY = input("Your key: ")
        print()

    # Record a sample voicefile
    print(Color.YELLOW + "[STEP 3]: " + Color.END + f"Please sing any note for {DURATION} seconds.")
    input("Press ENTER to start recording...")
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to complete
    print(Color.GREEN + "Recording complete." + Color.END)
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
    mic_stream = sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE, 
        callback=audio_callback)
    
    # THREAD 3: Plot Stream
    # animation = FuncAnimation(fig, update_plot, interval=10, blit=False)
    
    try:
        with mic_stream:
            while True:
                # plt.show()
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("**KEYBOARD INTERRUPT DETECTED**")
        mic_stream.stop()
        mic_stream.close()
        del midiout
        np.save('result.npy', NoteOn_seq)
        sys.exit(0)
    '''--------- threads end here ----------'''

except Exception as e:
    sys.exit()