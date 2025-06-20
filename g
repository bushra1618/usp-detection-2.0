import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightkurve import search_lightcurve
from astropy.timeseries import LombScargle, BoxLeastSquares
import os
import sys
import platform

sns.set(style='whitegrid')
plt.rcParams["figure.figsize"] = (10, 6)

target = "TIC 260004324"
sector = 1
plot_folder = "planet_plots_raw"

os.makedirs(plot_folder, exist_ok=True)

print(" Downloading light curve...")
search_result = search_lightcurve(target, mission="TESS", sector=sector, author="SPOC")
if not search_result:
    raise Exception(" No light curve found.")

# Download, clean, normalize — but no smoothing or binning
lc = search_result.download_all().stitch().remove_nans().remove_outliers(sigma=5)
lc = lc.normalize()  # Light normalization only

time = lc.time.value
flux = lc.flux

print(" Running Lomb-Scargle...")
min_freq = 1 / 5
max_freq = 1 / 0.05
frequency, power = LombScargle(time, flux).autopower(
    minimum_frequency=min_freq,
    maximum_frequency=max_freq,
    samples_per_peak=4
)
best_ls_period = 1 / frequency[np.argmax(power)]

plt.figure()
plt.plot(1 / frequency, power, 'k')
plt.axvline(best_ls_period, color='red', linestyle='--', label=f"Best LS Period: {best_ls_period:.4f} d")
plt.xlabel("Period (days)")
plt.ylabel("Power")
plt.title("Lomb-Scargle Periodogram (Raw)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_folder}/1_lomb_scargle_raw.png")
plt.close()
print(" LS periodogram saved.")

print(" Running Box Least Squares...")
periods = np.linspace(0.08, 0.3, 5000)
durations = 0.05 * periods
bls = BoxLeastSquares(time, flux)
results = bls.power(periods, durations)

best_idx = np.argmax(results.power)
best_period = results.period[best_idx]
best_t0 = results.transit_time[best_idx]
best_duration = results.duration[best_idx]
best_depth = results.depth[best_idx]

print("\n BLS Detection:")
print(f"• Period: {best_period:.6f} days (~{best_period*24:.2f} hours)")
print(f"• Duration: {best_duration*24:.2f} hours")
print(f"• Depth: {best_depth:.6f} (~{best_depth*1e6:.1f} ppm)")

plt.figure()
plt.plot(results.period, results.power, 'k')
plt.axvline(best_period, color='red', linestyle='--', label=f"P = {best_period:.4f} d")
plt.xlabel("Period (days)")
plt.ylabel("BLS Power")
plt.title("Box Least Squares Periodogram (Raw)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_folder}/2_bls_periodogram_raw.png")
plt.close()
print(" BLS periodogram saved.")

print(" Plotting phase-folded transit (raw)...")
phase = ((time - best_t0 + 0.5 * best_period) % best_period) / best_period
sort_idx = np.argsort(phase)

plt.figure()
plt.plot(phase[sort_idx], flux[sort_idx], 'ko', markersize=1.5, alpha=0.3, label='Raw Flux')

# Optional: overlay running median to guide the eye
bins = np.linspace(0, 1, 100)
digitized = np.digitize(phase, bins)
binned = [np.median(flux[digitized == i]) for i in range(1, len(bins))]
bin_centers = bins[:-1] + (bins[1] - bins[0]) / 2
plt.plot(bin_centers, binned, 'r-', linewidth=1.5, label='Running Median')

plt.xlabel("Orbital Phase")
plt.ylabel("Normalized Flux")
plt.title(f"Phase-Folded Light Curve @ P = {best_period:.4f} d (Raw Data)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_folder}/3_folded_transit_raw.png")
plt.close()
print(" Folded transit plot saved.")

print(" Opening plot folder...")
if platform.system() == "Windows":
    os.startfile(plot_folder)
else:
    print(f" Please manually check the folder: {os.path.abspath(plot_folder)}")

print("\n All done !")
