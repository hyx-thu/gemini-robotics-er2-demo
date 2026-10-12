

import sounddevice as sd
import soundfile as sf
import sys


def list_audio_devices():
    """
    Prints a formatted list of all available audio devices.
    """
    print("Available audio devices:")
    print("------------------------")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        # The '>' and '<' indicate the default input and output devices.
        is_default = ""
        if device['default_samplerate'] > 0:
            is_default = " (default)"
        
        print(f"Device ID: {i} | Name: {device['name']} | Channels: {device['max_output_channels']}{is_default}")
    print("------------------------")

def play_audio_on_device(filename, device_id):
    """
    Plays a .wav file through the specified audio device.

    Args:
        filename (str): The path to the .wav file.
        device_id (int): The ID of the audio device to use.
    """
    try:
        # Read the WAV file data and sampling rate
        data, fs = sf.read(filename, dtype='float32')
        
        # Check if the device exists and has output channels
        device_info = sd.query_devices(device=device_id)
        if device_info['max_output_channels'] == 0:
            print(f"Error: Device ID {device_id} has no output channels.")
            return

        print(f"\nPlaying '{filename}' on device ID {device_id}...")
        
        # Play the audio data on the selected device
        sd.play(data, samplerate=fs, device=device_id)
        
        # Wait for the playback to finish before exiting
        # sd.wait()
        
        print("Playback finished.")

    except sf.SoundFileError:
        print(f"Error: Could not read audio file '{filename}'. Make sure it's a valid WAV file.")
    except sd.PortAudioError as e:
        print(f"Error playing audio: {e}")
    except ValueError as e:
        print(f"Error: {e}. Please make sure you enter a valid device ID.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Specify the path to your WAV file here
    wav_file_path = "/home/boodoo/beeps/beep_beep2.wav"  # Change this to your file path

    # List all available audio devices
    list_audio_devices()

    try:
        # Prompt the user to enter the device ID
        # device_choice = int(input("Enter the Device ID you want to use for playback: "))
        device_choice = int()
        
        # Play the audio file on the selected device
        play_audio_on_device(wav_file_path, device_choice)
        
    except ValueError:
        print("Invalid input. Please enter a valid number for the device ID.")
    except KeyboardInterrupt:
        print("\nPlayback stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
