import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch


# =========================
# PARAMETER EEG
# =========================

SFREQ = 250


BANDS = {

    "delta": (1,4),

    "theta": (4,8),

    "alpha": (8,13),

    "beta":  (13,30)

}



# =========================
# FILTER
# =========================


def bandpass_filter(
        data,
        lowcut=1,
        highcut=40,
        fs=250,
        order=4
):

    nyquist = fs/2


    low = lowcut/nyquist

    high = highcut/nyquist


    b,a = butter(
        order,
        [low,high],
        btype="band"
    )


    return filtfilt(
        b,
        a,
        data,
        axis=0
    )



def notch_filter(
        data,
        freq=50,
        fs=250
):

    b,a = iirnotch(
        freq,
        30,
        fs
    )


    return filtfilt(
        b,
        a,
        data,
        axis=0
    )





# =========================
# PSD / BRAIN WAVE
# =========================


def calculate_band_power(
        signal,
        fs=250
):


    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=fs*2
    )


    power = {}


    for band,(low,high) in BANDS.items():


        idx = np.logical_and(
            freqs >= low,
            freqs <= high
        )


        power[band] = np.mean(
            psd[idx]
        )


    return power





# =========================
# FEATURE EXTRACTION
# =========================


def extract_features(raw_data):


    """
    input:
    raw_data =
    array:
    sample x channel


    output:

    features
    -> untuk model AI

    band_power
    -> untuk grafik

    clean_signal
    -> untuk grafik EEG
    """



    # =================
    # preprocessing
    # =================


    clean = bandpass_filter(
        raw_data,
        fs=SFREQ
    )


    clean = notch_filter(
        clean,
        fs=SFREQ
    )





    features = {}

    band_summary = {}



    # =================
    # per channel
    # =================


    for ch in range(
        clean.shape[1]
    ):


        signal = clean[:,ch]



        power = calculate_band_power(
            signal,
            SFREQ
        )



        for band,value in power.items():


            key = (
                f"ch{ch+1}_{band}"
            )


            features[key] = value



            if band not in band_summary:

                band_summary[band]=0


            band_summary[band]+=value





    # =================
    # normalisasi summary
    # =================


    for k in band_summary:

        band_summary[k] /= clean.shape[1]




    # =================
    # tambahan fitur ratio
    # =================


    alpha = band_summary["alpha"]

    beta = band_summary["beta"]


    features[
        "alpha_beta_ratio"
    ] = alpha/(beta+1e-8)




    return (

        features,

        band_summary,

        clean

    )