import numpy as np
from scipy.io.wavfile import write

def create_beep_boop_wav(filename="beep_beep2.wav", sample_rate=44100):
    """Generates a small WAV file with a sequence of beeps and boops."""

    # Set parameters for the audio
    duration_sec = 0.15
    silence_duration_sec = 0.05
    volume = 0.5  # Adjust volume, 0.0 to 1.0
    
    # Define frequencies for different "beeps" and "boops"
    freq_beep = 1000  # A high-pitched tone
    freq_boop = 440  # A lower-pitched tone
    freq_beep2 = 1200
    
    # Generate time array
    t = np.linspace(0., duration_sec, int(sample_rate * duration_sec))
    t2 = np.linspace(0., 0.25, int(sample_rate * 0.25))
    
    # Create the sine waves
    beep = volume * np.sin(2. * np.pi * freq_beep * t)
    boop = volume * np.sin(2. * np.pi * freq_boop * t)
    beep2 = volume * np.sin(2. * np.pi * freq_beep2 * t2)

    silence = np.zeros(int(sample_rate * silence_duration_sec))
    
    # Combine the sounds to create the sequence
    audio_data = np.concatenate([
        beep, silence,
        # boop, silence,
        beep2, silence,
        # boop, silence
    ])
    
    # Convert floating-point data to 16-bit integers
    # The `write` function requires integer data
    scaled_audio = np.int16(audio_data * (2**15 - 1))
    
    # Write the array to a WAV file
    write(filename, sample_rate, scaled_audio)
    
    print(f"File '{filename}' has been created.")

    # New combined code structure
def create_robot_sounds(sample_rate=44100):
    duration_sec = 0.15
    silence_duration_sec = 0.05
    volume = 0.5
    
    t = np.linspace(0., duration_sec, int(sample_rate * duration_sec))
    silence = np.zeros(int(sample_rate * silence_duration_sec))
    
    # Happy Beep
    happy_beep1 = volume * np.sin(2. * np.pi * 1000 * t)
    happy_beep2 = volume * np.sin(2. * np.pi * 1200 * t)
    happy_sequence = np.concatenate([happy_beep1, silence, happy_beep2])
    scaled_happy = np.int16(happy_sequence * (2**15 - 1))
    write("happy_beep.wav", sample_rate, scaled_happy)
    print("File 'happy_beep.wav' created.")

    # Acknowledgment Beep
    ack_beep = volume * np.sin(2. * np.pi * 550 * t)
    scaled_ack = np.int16(ack_beep * (2**15 - 1))
    write("ack_beep.wav", sample_rate, scaled_ack)
    print("File 'ack_beep.wav' created.")

    # Disapproval Boop
    dis_boop1 = volume * np.sin(2. * np.pi * 440 * t)
    dis_boop2 = volume * np.sin(2. * np.pi * 300 * t)
    dis_sequence = np.concatenate([dis_boop1, silence, dis_boop2])
    scaled_dis = np.int16(dis_sequence * (2**15 - 1))
    write("disapproval_boop.wav", sample_rate, scaled_dis)
    print("File 'disapproval_boop.wav' created.")

# Call the function to create all three files
create_robot_sounds()

# Call the function to create the WAV file
# create_beep_boop_wav()

