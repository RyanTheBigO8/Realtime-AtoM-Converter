import argparse
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import aubio
import librosa


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def parse_input():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
        help='input channels to plot (default: the first)')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-w', '--window', type=float, default=200, metavar='DURATION',
        help='visible time slot (default: %(default)s ms)')
    parser.add_argument(
        '-i', '--interval', type=float, default=10,
        help='minimum time between plot updates (default: %(default)s ms)')
    parser.add_argument(
        '-b', '--blocksize', type=int, help='block size (in samples)')
    parser.add_argument(
        '-r', '--samplerate', type=float, help='sampling rate of audio device')
    parser.add_argument(
        '-n', '--downsample', type=int, default=10, metavar='N',
        help='display every Nth sample (default: %(default)s)')
    args = parser.parse_args(remaining)
    if any(c < 1 for c in args.channels):
        parser.error('argument CHANNEL: must be >= 1')
    mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1

    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        print(device_info)
        args.samplerate = device_info['default_samplerate']

    print(f"device = {device_info['name']}")
    print(f"sample rate = {args.samplerate} samples/sec")
    print(f"display downsample = {args.downsample}")
    print(f"window length = {args.window}ms")
    print(f"update interval = {args.interval}ms")
    # print(f"# of samples in each window = {args.blocksize} samples")

    return args, parser

args, parser = parse_input()

q = queue.Queue()

buffer_size = 1024
hop_size = buffer_size // 2
midi_note_offset = 0
sample_rate = args.samplerate
energy_threshold = 0.003

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, flush=True)
    # Fancy indexing with mapping creates a (necessary!) copy:

    q.put(indata[::args.downsample])

    # Convert the input audio to mono
    mono_audio = librosa.to_mono(indata.T)

    # Get the pitches using librosa
    pitches, magnitudes = librosa.core.piptrack(y=mono_audio, sr=sample_rate, fmin=100, hop_length=hop_size, n_fft=1024)

    # check if the magnitude of the sound is high enough for pitch detection
    if np.mean(magnitudes) < energy_threshold:
        return

    # Extract the most prominent pitch for each frame
    indices = magnitudes.argmax(axis=0)
    pitch_values = pitches[indices, np.arange(pitches.shape[1])]

    # Get the median pitch as the final detected pitch
    median_pitch = np.median(pitch_values)

    # Convert pitch values to MIDI note numbers
    midi_note = round(librosa.core.hz_to_midi(median_pitch))

    print(f"Detected MIDI Note: {midi_note}", flush=True)


def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines


try:
    length = int(args.window * args.samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, len(args.channels)))

    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    if len(args.channels) > 1:
        ax.legend([f'channel {c}' for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax.axis((0, len(plotdata), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    fig.tight_layout(pad=0)

    # 'stream' is an int indicating the max. number of frames
    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback)

    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)

    with stream:
        plt.show()
        sd.sleep(1000000)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))