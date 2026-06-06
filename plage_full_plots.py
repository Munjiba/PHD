# ============================================================
# PLAGE COUNTS + ENERGY + MAGNETIC FLUX ANALYSIS
#
# BASED ON:
# 1. Your plage counts/energy code
# 2. Your SHARP flux/error methodology
#
# ============================================================

from astropy.io import fits
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scienceplots
import glob
import os
import math
import sunpy.map
from matplotlib.lines import Line2D
plt.style.use(['science','notebook'])

# ============================================================
# DIRECTORIES
# ============================================================

patch_suit_dir = "/home/munjiba/Documents/working/14056/aia_cont_suit/new_suit_cutout/update_time_cutout/"
patch_hmi_dir  = "/home/munjiba/Documents/working/14056/hmi_flux/"
patch_aia_dir  = "/home/munjiba/Documents/working/14056/aia_cont_suit/aia_smooth/reprojected/new_aia_cutout/"

full_suit_dir  = "/home/munjiba/Documents/working/14056/full_suit_data/reprojected_suit/"
full_hmi_dir   = "/home/munjiba/Documents/working/14056/hmi_magnetogram/"

# ============================================================
# FILES
# ============================================================

patch_suit_files = sorted(glob.glob(os.path.join(patch_suit_dir, "*NB03_cutout.fits")))
patch_hmi_files  = sorted(glob.glob(os.path.join(patch_hmi_dir, "*Br_cutout.fits")))
patch_aia_files  = sorted(glob.glob(os.path.join(patch_aia_dir, "*.fits")))

full_suit_files  = sorted(glob.glob(os.path.join(full_suit_dir, "*NB03.fits")))
full_hmi_files   = sorted(glob.glob(os.path.join(full_hmi_dir, "*.fits")))

# ============================================================
# CENTER PATCH
# ============================================================

x1, x2 = 1854, 2139
y1, y2 = 1967, 2234

# ============================================================
# CONSTANTS
# ============================================================

omega_pix = (0.7 * np.pi / (180 * 3600))**2

M = 3357.79
C = -1.98e6

sigma_M = 0.023
sigma_C = 96.69

radsindeg = np.pi / 180.0

# ============================================================
# FUNCTIONS
# ============================================================

def get_time_from_file(f):

    with fits.open(f) as hdul:

        for hdu in hdul:

            header = hdu.header

            for key in ['DATE-OBS','T_OBS','DATE','DATE-AVG']:

                if key in header:
                    return Time(header[key]).to_datetime()

    raise KeyError(f"No valid time keyword found in {f}")


def find_closest(target_time, file_list):

    min_diff = float('inf')
    closest_file = None

    for f in file_list:

        t = get_time_from_file(f)

        diff = abs((t - target_time).total_seconds())

        if diff < min_diff:
            min_diff = diff
            closest_file = f

    return closest_file


def compute_abs_flux(bz, bz_err, mask, rsun_ref, rsun_obs, cdelt1_arcsec):

    mask = mask & (~np.isnan(bz))

    if np.sum(mask) == 0:
        return np.nan, np.nan, 0

    sum_abs = np.sum(np.abs(bz[mask]))

    err_sq_sum = np.sum((bz_err[mask])**2)

    area_factor = (
        (cdelt1_arcsec**2)
        *
        (rsun_ref/rsun_obs)**2
        *
        100.0
        *
        100.0
    )

    flux = sum_abs * area_factor

    flux_err = np.sqrt(err_sq_sum) * area_factor

    return flux, flux_err, np.sum(mask)


def compute_total_field(bz, bz_err, mask):

    mask = mask & (~np.isnan(bz))

    if np.sum(mask) == 0:
        return np.nan, np.nan

    total_field = np.sum(np.abs(bz[mask]))

    total_field_err = np.sqrt(np.sum((bz_err[mask])**2))

    return total_field, total_field_err


def save_plot(times, vals, errs, ylabel, filename):

    plt.figure(figsize=(10,6))

    plt.errorbar(
        times,
        vals,
        yerr=errs,
        fmt='o',
        capsize=3,
        label=ylabel
    )

    plt.xlabel("")
    #plt.xlabel("Date", fontsize=16)
    plt.ylabel(ylabel, fontsize=15)

    plt.xticks(color='white')
    plt.yticks(fontsize=13)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(filename, dpi=300)

    plt.close()


def save_correlation_plot(
    x,
    y,
    xerr,
    yerr,
    xlabel,
    ylabel,
    filename
):

    plt.figure(figsize=(10,6))
    """
    plt.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt='o',
        capsize=3
    )
    """

    y_plot = plt.errorbar(
        x,
        y,
        yerr=yerr,
        color = 'blue',
        fmt='o',
        capsize=3
    )

    x_plot = plt.errorbar(
        x,
        y,
        xerr=xerr,
        color = 'blue',
        fmt='o',
        capsize=3
    )




    plt.xlabel('')
    plt.ylabel(ylabel, fontsize=15)
    plt.legend(
        [y_plot, x_plot],
        [ylabel, xlabel],
        fontsize=11,
        loc='best'
    )

    plt.tight_layout()


    plt.savefig(filename, dpi=300)

    plt.close()


def save_dual_axis_plot(
    times,
    left_vals,
    left_errs,
    right_vals,
    right_errs,
    left_ylabel,
    right_ylabel,
    filename
):

    fig, ax1 = plt.subplots(figsize=(10,6))

    # -----------------------------------------
    # LEFT AXIS
    # -----------------------------------------

    ax1.errorbar(
        times,
        left_vals,
        yerr=left_errs,
        fmt='o',
        capsize=3,
        color='blue',
        ecolor='blue',
        markersize=5,
        label=left_ylabel
    )

    ax1.set_xlabel("")

    ax1.set_ylabel(
        left_ylabel,
        fontsize=15,
        color='blue'
    )

    ax1.tick_params(
        axis='y',
        colors='blue'
    )

    ax1.spines['left'].set_color('blue')

    # -----------------------------------------
    # RIGHT AXIS
    # -----------------------------------------

    ax2 = ax1.twinx()

    ax2.errorbar(
        times,
        right_vals,
        yerr=right_errs,
        fmt='s',
        capsize=3,
        color='red',
        ecolor='red',
        markersize=5,
        label=right_ylabel
    )

    ax2.set_ylabel(
        right_ylabel,
        fontsize=15,
        color='red'
    )

    ax2.tick_params(
        axis='y',
        colors='red'
    )

    ax2.spines['right'].set_color('red')
    lines1, labels1 = ax1.get_legend_handles_labels()

    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper right',
        fontsize=11
    )
    """
    locator = mdates.AutoDateLocator()

    ax1.xaxis.set_major_locator(locator)

    ax1.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )

    fig.autofmt_xdate()
    """
    #ax1.set_xticks(color='white')
    ax1.tick_params(
        axis='x',
        labelbottom=False
    )
    plt.tight_layout()

    
    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()

# ============================================================
# STORAGE
# ============================================================

times = []

total_counts_arr = []
total_counts_err_arr = []

mean_counts_arr = []
mean_counts_err_arr = []

total_energy_arr = []
total_energy_err_arr = []

mean_energy_arr = []
mean_energy_err_arr = []

total_flux_arr = []
total_flux_err_arr = []

mean_flux_arr = []
mean_flux_err_arr = []

total_field_arr = []
total_field_err_arr = []

mean_field_arr = []
mean_field_err_arr = []

# ============================================================
# MAIN LOOP
# ============================================================

for suit_patch_file in patch_suit_files:

    try:

        # ====================================================
        # SUIT PATCH
        # ====================================================

        hdul = fits.open(suit_patch_file)

        suit_patch = hdul[0].data.astype(float)

        header = hdul[0].header

        time_patch = get_time_from_file(suit_patch_file)

        exptime = header.get("EXPTIME",1)

        if exptime != 0:
            suit_patch = suit_patch / exptime

        suit_patch = np.nan_to_num(suit_patch, nan=0.0)

        # ====================================================
        # MATCH FILES
        # ====================================================

        hmi_patch_file = find_closest(
            time_patch,
            patch_hmi_files
        )

        aia_patch_file = find_closest(
            time_patch,
            patch_aia_files
        )

        # ====================================================
        # HMI
        # ====================================================

        bz_map = sunpy.map.Map(hmi_patch_file)

        bz = bz_map.data.astype(float)

        header_bz = bz_map.meta

        base = os.path.basename(hmi_patch_file).replace(".Br_cutout.fits","")

        bz_err_file = next(
            (
                f for f in glob.glob(
                    os.path.join(
                        patch_hmi_dir,
                        "*Br_err*.fits"
                    )
                )
                if base in os.path.basename(f)
            ),
            None
        )

        if bz_err_file is None:
            raise FileNotFoundError(
                f"No Br_err file found for {base}"
            )

        bz_err = fits.getdata(bz_err_file).astype(float)


        magnetogram_file = next(
            (
                f for f in glob.glob(
                    os.path.join(
                        patch_hmi_dir,
                        "*.magnetogram_cutout*.fits"
                    )
                )
                if base in os.path.basename(f)
            ),
            None
        )
        if magnetogram_file is None:
            raise FileNotFoundError(
                f"No Br_err file found for {base}"
            )

        magnetogram = fits.getdata(magnetogram_file).astype(float)


        # ====================================================
        # AIA PATCH
        # ====================================================

        aia_patch = fits.getdata(aia_patch_file).astype(float)

        aia_patch = np.nan_to_num(aia_patch, nan=0.0)

        # ====================================================
        # QUIET SUN MASK
        # ====================================================

        mask_patch = np.abs(magnetogram) < 20

        if np.sum(mask_patch) == 0:
            continue

        aia_qs_values = aia_patch[mask_patch]

        aia_qs_patch = np.median(aia_qs_values)

        aia_qs_std = np.std(aia_qs_values)

        if aia_qs_patch == 0:
            continue

        # ====================================================
        # PLAGE MASK
        # ====================================================

        plage_mask = aia_patch > (aia_qs_patch + 3 * aia_qs_std)

        #plage_mask = aia_patch > (1.09 * aia_qs_patch)

        if np.sum(plage_mask) == 0:
            continue

        # ====================================================
        # SUIT PLAGE
        # ====================================================

        suit_plage = suit_patch[plage_mask]

        total_counts_raw = np.sum(suit_plage)

        N_pixels = suit_plage.size

        mean_counts_raw = total_counts_raw / N_pixels

        # ====================================================
        # FULL SUIT
        # ====================================================

        full_suit_file = find_closest(
            time_patch,
            full_suit_files
        )

        hdul_full = fits.open(full_suit_file)

        full_suit = hdul_full[0].data.astype(float)

        header_full = hdul_full[0].header

        exptime_full = header_full.get("EXPTIME",1)

        if exptime_full != 0:
            full_suit = full_suit / exptime_full

        full_suit = np.nan_to_num(full_suit, nan=0.0)

        full_center = full_suit[y1:y2, x1:x2]

        # ====================================================
        # FULL HMI
        # ====================================================

        full_hmi_file = find_closest(
            time_patch,
            full_hmi_files
        )

        full_hmi = fits.getdata(full_hmi_file)

        full_hmi_center = full_hmi[y1:y2, x1:x2]

        mask_full = np.abs(full_hmi_center) < 20

        if np.sum(mask_full) == 0:
            continue

        qs_center = np.median(full_center[mask_full])

        suit_qs_patch = np.median(suit_patch[mask_patch])

        if suit_qs_patch == 0:
            continue

        # ====================================================
        # COUNTS
        # ====================================================

        correction_factor = qs_center / suit_qs_patch

        A_total = total_counts_raw * correction_factor

        A_mean = mean_counts_raw * correction_factor

        # ====================================================
        # ERRORS
        # ====================================================

        sigma_tc = np.sqrt(total_counts_raw)

        sigma_mc = sigma_tc / N_pixels

        sigma_qc = (
            np.std(full_center[mask_full])
            /
            np.sqrt(np.sum(mask_full))
        )

        sigma_qp = (
            np.std(suit_patch[mask_patch])
            /
            np.sqrt(np.sum(mask_patch))
        )

        sigma_A_total = A_total * np.sqrt(
            (sigma_tc/total_counts_raw)**2
            +
            (sigma_qc/qs_center)**2
            +
            (sigma_qp/suit_qs_patch)**2
        )

        sigma_A_mean = A_mean * np.sqrt(
            (sigma_mc/mean_counts_raw)**2
            +
            (sigma_qc/qs_center)**2
            +
            (sigma_qp/suit_qs_patch)**2
        )

        # ====================================================
        # ENERGY
        # ====================================================

        total_energy = (
            (
                A_total * M
                +
                C * N_pixels
            )
            *
            omega_pix
        )

        sigma_B_total = np.sqrt(
            (M*sigma_A_total)**2
            +
            (A_total*sigma_M)**2
            +
            (N_pixels*sigma_C)**2
        )

        sigma_total_energy = omega_pix * sigma_B_total

        mean_energy = (
            (
                A_mean * M
                +
                C
            )
            *
            omega_pix
        )

        sigma_B_mean = np.sqrt(
            (M*sigma_A_mean)**2
            +
            (A_mean*sigma_M)**2
            +
            (sigma_C)**2
        )

        sigma_mean_energy = omega_pix * sigma_B_mean

        # ====================================================
        # FLUX
        # ====================================================

        rsun_ref = header_bz.get('rsun_ref')
        rsun_obs = header_bz.get('rsun_obs')
        dsun_obs = header_bz.get('dsun_obs')
        cdelt1 = header_bz.get('cdelt1')

        cdelt1_arcsec = (
            math.atan(
                (
                    rsun_ref
                    *
                    cdelt1
                    *
                    radsindeg
                )
                /
                dsun_obs
            )
            *
            (1/radsindeg)
            *
            3600.0
        )

        total_flux, total_flux_err, npix_flux = compute_abs_flux(
            bz,
            bz_err,
            plage_mask,
            rsun_ref,
            rsun_obs,
            cdelt1_arcsec
        )

        total_field, total_field_err = compute_total_field(
            bz,
            bz_err,
            plage_mask
        )

        mean_flux = total_flux / npix_flux

        mean_flux_err = total_flux_err / npix_flux

        mean_field = total_field / npix_flux

        mean_field_err = total_field_err / npix_flux

        # ====================================================
        # STORE
        # ====================================================

        times.append(time_patch)

        total_counts_arr.append(A_total)
        total_counts_err_arr.append(sigma_A_total)

        mean_counts_arr.append(A_mean)
        mean_counts_err_arr.append(sigma_A_mean)

        total_energy_arr.append(total_energy)
        total_energy_err_arr.append(sigma_total_energy)

        mean_energy_arr.append(mean_energy)
        mean_energy_err_arr.append(sigma_mean_energy)

        total_flux_arr.append(total_flux)
        total_flux_err_arr.append(total_flux_err)

        mean_flux_arr.append(mean_flux)
        mean_flux_err_arr.append(mean_flux_err)

        total_field_arr.append(total_field)
        total_field_err_arr.append(total_field_err)

        mean_field_arr.append(mean_field)
        mean_field_err_arr.append(mean_field_err)

        print(f"Processed: {time_patch}")

    except Exception as e:

        print(f"ERROR: {e}")

# ============================================================
# TIME SERIES PLOTS
# ============================================================

save_plot(times,total_counts_arr,total_counts_err_arr,
"Total Plage Counts (DN)",
"total_counts_dn_plage.png")

save_plot(times,mean_counts_arr,mean_counts_err_arr,
"Mean Plage Counts (DN)",
"mean_counts_dn_plage.png")

save_plot(times,total_energy_arr,total_energy_err_arr,
r"Total Plage Energy ($ergs/s/cm^2$)",
"total_energy_plage.png")

save_plot(times,total_field_arr,total_field_err_arr,
"Total Plage Magnetic Field (G)",
"total_field_plage.png")

save_plot(times,mean_field_arr,mean_field_err_arr,
"Mean Plage Magnetic Field (G)",
"mean_field_plage.png")

save_plot(times,mean_energy_arr,mean_energy_err_arr,
r"Mean Plage Energy ($ergs/s/cm^2$)",
"mean_energy_plage.png")

save_plot(times,total_flux_arr,total_flux_err_arr,
"Total Unsigned Magnetic Flux (Mx)",
"total_flux_plage.png")

save_plot(times,mean_flux_arr,mean_flux_err_arr,
"Mean Unsigned Magnetic Flux (Mx)",
"mean_flux_plage.png")

# ============================================================
# CORRELATION PLOTS
# ============================================================

save_correlation_plot(
    total_flux_arr,
    total_counts_arr,
    total_flux_err_arr,
    total_counts_err_arr,
    "Total Unsigned Flux (Mx)",
    "Total Counts (DN)",
    "total_counts_vs_total_flux_plage.png"
)

save_correlation_plot(
    total_field_arr,
    total_counts_arr,
    total_field_err_arr,
    total_counts_err_arr,
    "Total Magnetic Field (G)",
    "Total Counts (DN)",
    "total_counts_vs_total_field_plage.png"
)

save_correlation_plot(
    mean_flux_arr,
    mean_counts_arr,
    mean_flux_err_arr,
    mean_counts_err_arr,
    "Mean Unsigned Flux (Mx)",
    "Mean Counts (DN)",
    "mean_counts_vs_mean_flux_plage.png"
)

save_correlation_plot(
    mean_field_arr,
    mean_counts_arr,
    mean_field_err_arr,
    mean_counts_err_arr,
    "Mean Magnetic Field (G)",
    "Mean Counts (DN)",
    "mean_counts_vs_mean_field_plage.png"
)

save_correlation_plot(
    total_flux_arr,
    total_energy_arr,
    total_flux_err_arr,
    total_energy_err_arr,
    "Total Unsigned Flux (Mx)",
    r"Total Energy ($ergs/s/cm^2$)",
    "total_energy_vs_total_flux_plage.png"
)

save_correlation_plot(
    total_field_arr,
    total_energy_arr,
    total_field_err_arr,
    total_energy_err_arr,
    "Total Magnetic Field (G)",
    r"Total Energy ($ergs/s/cm^2$)",
    "total_energy_vs_total_field_plage.png"
)

save_correlation_plot(
    mean_flux_arr,
    mean_energy_arr,
    mean_flux_err_arr,
    mean_energy_err_arr,
    "Mean Unsigned Flux (Mx)",
    r"Mean Energy ($ergs/s/cm^2$)",
    "mean_energy_vs_mean_flux_plage.png"
)

save_correlation_plot(
    mean_field_arr,
    mean_energy_arr,
    mean_field_err_arr,
    mean_energy_err_arr,
    "Mean Magnetic Field (G)",
    r"Mean Energy ($ergs/s/cm^2$)",
    "mean_energy_vs_mean_field_plage.png"
)

# ============================================================
# DUAL AXIS PLOTS
# ============================================================

save_dual_axis_plot(
    times,
    total_counts_arr,
    total_counts_err_arr,
    total_flux_arr,
    total_flux_err_arr,
    "Total Counts (DN)",
    "Total Unsigned Flux (Mx)",
    "total_counts_total_flux_time_plage.png"
)

save_dual_axis_plot(
    times,
    total_counts_arr,
    total_counts_err_arr,
    total_field_arr,
    total_field_err_arr,
    "Total Counts (DN)",
    "Total Magnetic Field (G)",
    "total_counts_total_field_time_plage.png"
)

save_dual_axis_plot(
    times,
    mean_counts_arr,
    mean_counts_err_arr,
    mean_flux_arr,
    mean_flux_err_arr,
    "Mean Counts (DN)",
    "Mean Unsigned Flux (Mx)",
    "mean_counts_mean_flux_time_plage.png"
)

save_dual_axis_plot(
    times,
    mean_counts_arr,
    mean_counts_err_arr,
    mean_field_arr,
    mean_field_err_arr,
    "Mean Counts (DN)",
    "Mean Magnetic Field (G)",
    "mean_counts_mean_field_time_plage.png"
)

save_dual_axis_plot(
    times,
    total_energy_arr,
    total_energy_err_arr,
    total_flux_arr,
    total_flux_err_arr,
    r"Total Energy ($ergs/s/cm^2$)",
    "Total Unsigned Flux (Mx)",
    "total_energy_total_flux_time_plage.png"
)

save_dual_axis_plot(
    times,
    total_energy_arr,
    total_energy_err_arr,
    total_field_arr,
    total_field_err_arr,
    r"Total Energy ($ergs/s/cm^2$)",
    "Total Magnetic Field (G)",
    "total_energy_total_field_time_plage.png"
)

save_dual_axis_plot(
    times,
    mean_energy_arr,
    mean_energy_err_arr,
    mean_flux_arr,
    mean_flux_err_arr,
    r"Mean Energy ($ergs/s/cm^2$)",
    "Mean Unsigned Flux (Mx)",
    "mean_energy_mean_flux_time_plage.png"
)

save_dual_axis_plot(
    times,
    mean_energy_arr,
    mean_energy_err_arr,
    mean_field_arr,
    mean_field_err_arr,
    r"Mean Energy ($ergs/s/cm^2$)",
    "Mean Magnetic Field (G)",
    "mean_energy_mean_field_time_plage.png"
)

print("==========================================")
print("ALL 22 PLAGE PLOTS CREATED SUCCESSFULLY")
print("==========================================")