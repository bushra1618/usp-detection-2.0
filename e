import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightkurve import search_lightcurvefile
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter
from tqdm import tqdm
import os
import platform

sns.set(style='whitegrid')
plt.rcParams["figure.figsize"] = (10, 6)

SECTOR = 70  # <-- 
MIN_PERIOD = 0.05  # 1.2 hours
MAX_PERIOD = 1.0   # 24 hours
BIN_FACTOR = 5
PLOT_FOLDER = f"usp_candidates_sector{SECTOR}"
os.makedirs(PLOT_FOLDER, exist_ok=True)

print(f"Finding targets in TESS Sector {SECTOR}...")
# Get all available light curve files for the chosen sector
results = search_lightcurvefile(mission="TESS", sector=SECTOR)
tic_ids = [r.target_name for r in results[:20]]  # <-- Small sample for light PC

print(f"Found {len(tic_ids)} targets with light curves in sector {SECTOR}. Starting USP search...")

for tic in tqdm(tic_ids, desc="Searching for USPs"):
    try:
        lcfile = search_lightcurvefile(tic, mission="TESS", sector=SECTOR).download()
        if lcfile is None:
            continue
        lc = lcfile.PDCSAP_FLUX.remove_nans().remove_outliers(sigma=5).flatten().normalize()
        if len(lc.flux) > 5000:
            time = lc.time.value[::BIN_FACTOR]
            flux = lc.flux[::BIN_FACTOR]
        else:
            time = lc.time.value
            flux = lc.flux

        wl = 101 if len(flux) > 101 else len(flux)//2*2+1
        flux = savgol_filter(flux, window_length=wl, polyorder=2)

        # Box Least Squares periodogram for USPs
        periods = np.linspace(MIN_PERIOD, MAX_PERIOD, 4000)
        durations = 0.05 * periods
        bls = BoxLeastSquares(time, flux)
        results = bls.power(periods, durations)
        best_idx = np.argmax(results.power)
        best_period = results.period[best_idx]
        best_power = results.power[best_idx]
        best_t0 = results.transit_time[best_idx]
        best_depth = results.depth[best_idx]

        # Criteria for candidate: strong BLS peak, negative (dimming) depth, not at search edge
        if best_power > 0.1 and best_depth < 0 and MIN_PERIOD < best_period < MAX_PERIOD:
            phase = ((time - best_t0 + 0.5 * best_period) % best_period) / best_period
            sort_idx = np.argsort(phase)
            bins = np.linspace(0, 1, 80)
            digitized = np.digitize(phase, bins)
            binned = [np.median(flux[digitized == i]) for i in range(1, len(bins))]
            bin_centers = bins[:-1] + (bins[1] - bins[0]) / 2
            # Plot and save only the candidate
            plt.figure()
            plt.plot(phase[sort_idx], flux[sort_idx], 'ko', markersize=2, alpha=0.4, label='Data')
            plt.plot(bin_centers, binned, 'r-', label='Binned')
            plt.xlabel("Orbital Phase")
            plt.ylabel("Normalized Flux")
            plt.title(f"TIC {tic}: Candidate USP @ P = {best_period:.4f} d")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{PLOT_FOLDER}/TIC_{tic}_USP.png")
            plt.close()
            print(f"\nCANDIDATE FOUND! TIC {tic} at P={best_period:.4f} d, Power={best_power:.2f}")
            print(f"  Depth: {best_depth:.1e} | t0: {best_t0:.4f}")
    except Exception as e:
        print(f"Error for TIC {tic}: {e}")

print("Done! Check your usp_candidates_sector{SECTOR}/ folder for candidates.")
if platform.system() == "Windows":
    os.startfile(PLOT_FOLDER)
else:
    print(f"Open: {os.path.abspath(PLOT_FOLDER)}")
