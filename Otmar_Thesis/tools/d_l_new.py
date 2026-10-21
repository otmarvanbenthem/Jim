import numpy as np

def new_dl(old_dl,old_snr,new_snr):
    dl = (old_snr * old_dl)/new_snr
    return dl

print("SNR 20Hz ET 2L:", new_dl(500,44.65,30))
print("SNR 5Hz ET 2L:", new_dl(500,64.6,30))
