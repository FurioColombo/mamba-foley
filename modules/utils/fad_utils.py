import os
import shutil
import soundfile as sf
import numpy as np
from utils.file_system import ProjectPaths

SAMPLE_RATE = 16000


def add_noise(data, stddev):
  """Adds Gaussian noise to the samples.
  Args:
    data: 1d Numpy array containing floating point samples. Not necessarily
      normalized.
    stddev: The standard deviation of the added noise.
  Returns:
     1d Numpy array containing the provided floating point samples with added
     Gaussian noise.
  Raises:
    ValueError: When data is not a 1d numpy array.
  """
  if len(data.shape) != 1:
    raise ValueError("expected 1d numpy array.")
  max_value = np.amax(np.abs(data))
  num_samples = data.shape[0]
  gauss = np.random.normal(0, stddev, (num_samples)) * max_value
  return data + gauss


def gen_sine_wave(freq=600,
                  length_seconds=6,
                  sample_rate=SAMPLE_RATE,
                  noise_param=None):
  """Creates sine wave of the specified frequency, sample_rate and length."""
  t = np.linspace(0, length_seconds, int(length_seconds * sample_rate))
  samples = np.sin(2 * np.pi * t * freq)
  if noise_param:
    samples = add_noise(samples, noise_param)
  return np.asarray(2**15 * samples, dtype=np.int16)

def test_fad(frechet):
    LENGTH_IN_SECONDS = 4
    # test FAD is working
    def gen_audio_folder_path(path: str=''):
        encodec_folder_path = ProjectPaths.encodec_folder_path
        new_folder = os.path.join(encodec_folder_path, 'temp_test_audio')
        os.makedirs(new_folder, exist_ok=True)
        new_path = os.path.join(new_folder, path)
        return new_path

    for target, count, param in [('background', 10, None), ("test1", 5, 0.0001), ("test2", 10, 0.0000001)]:
        target = gen_audio_folder_path(target)
        os.makedirs(target, exist_ok=True)
        frequencies = np.linspace(100, 1000, count).tolist()
        for freq in frequencies:
            samples = gen_sine_wave(freq, LENGTH_IN_SECONDS, SAMPLE_RATE, noise_param=param)
            filename = os.path.join(target, "sin_%.0f.wav" % freq)
            # print("Creating: %s with %i samples." % (filename, samples.shape[0]))
            # print(os.path.abspath(filename))
            sf.write(filename, samples, SAMPLE_RATE, "PCM_24")

    fad_score = frechet.score(
        gen_audio_folder_path("background"),
        gen_audio_folder_path("test1"),
        dtype="float32"
    )
    print("FAD score test 1: %.8f" % fad_score)

    fad_score = frechet.score(
        gen_audio_folder_path("background"),
        gen_audio_folder_path("test2"),
        dtype="float32"
    )
    print("FAD score test 2: %.8f" % fad_score)

    shutil.rmtree(fad_encoded_emb_folder_path("background"))
    shutil.rmtree(fad_encoded_emb_folder_path("test1"))
    shutil.rmtree(fad_encoded_emb_folder_path("test2"))
