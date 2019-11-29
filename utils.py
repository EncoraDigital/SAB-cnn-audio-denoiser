import numpy as np
import pickle
import librosa
import sounddevice as sd
import tensorflow as tf


def inverse_stft_transform(stft_features, window_length, overlap):
    return librosa.istft(stft_features, win_length=window_length, hop_length=overlap)


def revert_features_to_audio(features, phase, window_length, overlap, cleanMean=None, cleanStd=None):
    # scale the outpus back to the original range
    if cleanMean and cleanStd:
        features = cleanStd * features + cleanMean

    phase = np.transpose(phase, (1, 0))
    features = np.squeeze(features)
    features = features * np.exp(1j * phase)  # that fixes the abs() ope previously done

    features = np.transpose(features, (1, 0))
    return inverse_stft_transform(features, window_length=window_length, overlap=overlap)


def play(audio, sample_rate):
    # ipd.display(ipd.Audio(data=audio, rate=sample_rate))  # load a local WAV file
    sd.play(audio, sample_rate, blocking=True)


def add_noise_to_clean_audio(clean_audio, noise_signal):
    if len(clean_audio) >= len(noise_signal):
        # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
        while len(clean_audio) >= len(noise_signal):
            noise_signal = np.append(noise_signal, noise_signal)

    ## Extract a noise segment from a random location in the noise file
    ind = np.random.randint(0, noise_signal.size - clean_audio.size)

    noiseSegment = noise_signal[ind: ind + clean_audio.size]

    speech_power = np.sum(clean_audio ** 2)
    noise_power = np.sum(noiseSegment ** 2)
    noisyAudio = clean_audio + np.sqrt(speech_power / noise_power) * noiseSegment
    return noisyAudio

def read_audio(filepath, sample_rate, normalize=True):
    audio, sr = librosa.load(filepath, sr=sample_rate)
    if normalize is True:
        div_fac = 1 / np.max(np.abs(audio)) / 3.0
        audio = audio * div_fac
        # audio = librosa.util.normalize(audio)
    return audio, sr


def prepare_input_features(stft_features, numSegments, numFeatures):
    noisySTFT = np.concatenate([stft_features[:, 0:numSegments - 1], stft_features], axis=1)
    stftSegments = np.zeros((numFeatures, numSegments, noisySTFT.shape[1] - numSegments + 1))

    for index in range(noisySTFT.shape[1] - numSegments + 1):
        stftSegments[:, :, index] = noisySTFT[:, index:index + numSegments]
    return stftSegments


def get_input_features(predictorsList):
    predictors = []
    for noisy_stft_mag_features in predictorsList:
        # For CNN, the input feature consisted of 8 consecutive noisy
        # STFT magnitude vectors of size: 129 × 8,
        # TODO: duration: 100ms
        inputFeatures = prepare_input_features(noisy_stft_mag_features)
        # print("inputFeatures.shape", inputFeatures.shape)
        predictors.append(inputFeatures)

    return predictors


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_tf_feature(noise_stft_mag_features, clean_stft_magnitude, noise_stft_phase):
    noise_stft_mag_features = noise_stft_mag_features.astype(np.float32).tostring()
    clean_stft_magnitude = clean_stft_magnitude.astype(np.float32).tostring()
    noise_stft_phase = noise_stft_phase.astype(np.float32).tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'noise_stft_phase': _bytes_feature(noise_stft_phase),
        'noise_stft_mag_features': _bytes_feature(noise_stft_mag_features),
        'clean_stft_magnitude': _bytes_feature(clean_stft_magnitude)}))
    return example
