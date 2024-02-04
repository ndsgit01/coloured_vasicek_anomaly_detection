"""
Functions to generate coloured noise time series, and estimate covariance matrix of noise series
"""

import numpy as np


"""
Reference:
- https://stackoverflow.com/a/67127726
"""

def noise_series(n, freq_mod = lambda f: 1):
    # White noise - samples from standard gaussian
    rng = np.random.default_rng()
    white_noise = rng.standard_normal(n)
    # Fourier transform white noise
    X_white = np.fft.rfft(white_noise)
    # sample frequency
    S = np.fft.rfftfreq(n)
    # Frquency modification
    S = freq_mod(S)
    # Normalize to preserve energy of white noise
    S = S / np.sqrt(np.mean(S**2))
    # Modify fourier coefficients
    X_shaped = X_white * S
    # Apply inverse fourier transform
    return np.fft.irfft(X_shaped);

def freq_modifier(f):
    return lambda n: noise_series(n, f)

@freq_modifier
def brown_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@freq_modifier
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))

@freq_modifier
def white_noise(f):
    return 1;

@freq_modifier
def blue_noise(f):
    return np.sqrt(f);

@freq_modifier
def violet_noise(f):
    return f;
    

"""
Reference:
https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices#Estimation_in_a_general_context
"""    
def generate_noise_samples(noise_func, noise_length, sample_size):
    noise_samples = np.zeros((noise_length, sample_size))
    for i in range(sample_size):
        noise_samples[:, i] = noise_func(noise_length)
    return noise_samples
    
def estimate_covariance_matrix(noise_func, noise_length=100, sample_size=100000):
    noise_samples = generate_noise_samples(noise_func, noise_length, sample_size)
    return np.cov(noise_samples) + (1e-10)*np.eye(noise_length)