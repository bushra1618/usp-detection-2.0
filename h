import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightkurve import search_lightcurve
import os
import platform

sns.set(style='whitegrid')
plt.rcParams["figure.figsize"] = (10, 6)


target = "TIC 260004324"
sector = 1
period = 3.814        # Known orbital period of LHS 1815b
t0 = 1325.5           # Approximate mid-transit time (BJD - 2457000)
window = 0.5          # Phase window around transit (±0.5 period)
nphase_bins = 100     # Number of phase bins per transit

plot_folder = "transit_stack"
os.makedirs(plot_folder, exist_ok=True)

print(" Downloading light curve...")
search_result = search_lightcurve(target, mission="TESS", sector=sector, author="SPOC")
if not search_result:
    raise Exception(" No light curve found.")

# Download and clean
lc = search_result.download_all().stitch().remove_nans().remove_outliers(sigma=5)
lc = lc.normalize()

# Convert to plain NumPy arrays without units
time = np.asarray(lc.time.value, dtype=float)
flux = np.asarray(lc.flux, dtype=float)

# ---- Extract transits ---- #
n_orbits = int((time[-1] - time[0]) / period)
phased_fluxes = []

for i in range(n_orbits):
    t_center = t0 + i * period
    mask = np.abs(time - t_center) < (window * period)
    if np.sum(mask) < 30:
        continue

    phase = (time[mask] - t_center) / period
    f = flux[mask]

    # Ensure phase and flux are clean float arrays
    phase = np.asarray(phase, dtype=float)
    f = np.asarray(f, dtype=float)

    valid_mask = np.isfinite(phase) & np.isfinite(f)
    if np.sum(valid_mask) < 10:
        continue

    bins = np.linspace(-window, window, nphase_bins)

    interp_flux = np.interp(
        bins,
        phase[valid_mask],
        f[valid_mask]
    )

    phased_fluxes.append(interp_flux)

phased_fluxes = np.array(phased_fluxes)

# ---- Plot heatmap ---- #
plt.figure(figsize=(10, 6))
vmin = np.nanpercentile(phased_fluxes, 5)
vmax = np.nanpercentile(phased_fluxes, 95)

plt.imshow(phased_fluxes, aspect='auto', cmap='viridis',
           extent=[-window, window, 0, phased_fluxes.shape[0]],
           origin='lower', interpolation='none', vmin=vmin, vmax=vmax)

plt.colorbar(label='Normalized Flux')
plt.xlabel("Orbital Phase (cycles from mid-transit)")
plt.ylabel("Transit Number")
plt.title("LHS 1815b Transit Stack: Orbit-to-Orbit Variation")
plt.tight_layout()
plt.savefig(f"{plot_folder}/stacked_transits_heatmap.png")
plt.close()
print(" Transit stack heatmap saved!")

# ---- Overlay all transits as lines ---- #
plt.figure(figsize=(10, 6))
bins = np.linspace(-window, window, nphase_bins)
for row in phased_fluxes:
    plt.plot(bins, row, color='gray', alpha=0.2)

plt.plot(bins, np.nanmedian(phased_fluxes, axis=0), 'r-', label='Median Transit')
plt.xlabel("Orbital Phase")
plt.ylabel("Normalized Flux")
plt.title("All Transits Overlayed")
plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_folder}/stacked_transits_lines.png")
plt.close()
print("Overlaid transit lines plot saved!")


print(" Opening plot folder...")
if platform.system() == "Windows":
    os.startfile(plot_folder)
else:
    print(f"Please check the folder: {os.path.abspath(plot_folder)}")

print("\n Transit stacking complete.")

