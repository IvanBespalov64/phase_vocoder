import librosa
import soundfile

import numpy as np
from scipy.fftpack import fft, ifft

import sys

def stretch(input_filename, output_filename, t_scale = 1., window_len=512):
    
    s, fs = librosa.load(input_filename)
    
    audio_length = s.size

    window_len = 512
    synt_len = window_len // 4

    out_length = int(audio_length / t_scale + window_len)

    phi  = np.zeros(window_len)
    out = np.zeros(window_len, dtype=complex)
    sigout = np.zeros(out_length)

    expected_ph = 2 * np.pi * synt_len / window_len * np.arange(0, window_len).T

    win = np.hanning(window_len)
    pout = 0
    pstretch = 0

    while pstretch < audio_length-(window_len+synt_len):

        p = int(pstretch)

        prev_spec =  fft(win*s[p:p+window_len])
        cur_spec =  fft(win*s[p+synt_len:p+window_len+synt_len])

        cur_magn = np.abs(cur_spec)

        phi += (np.angle(cur_spec) - np.angle(prev_spec)) - expected_ph
        phi =  ((-phi + np.pi) % (2.0 * np.pi) - np.pi) + expected_ph      

        out.real, out.imag = np.cos(phi), np.sin(phi)

        out *= cur_magn
        sigout[pout:pout+window_len] += win * ifft(out).real

        pout += synt_len
        pstretch += synt_len*t_scale
        
    soundfile.write(output_filename, sigout, fs)

if __name__ == '__main__':
    stretch(sys.argv[1], sys.argv[2], t_scale=float(sys.argv[3]))
