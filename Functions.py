import abtem
import numpy as np
import ase
import ase.spacegroup
from ase.spacegroup import crystal
from abtem.visualize import show_atoms
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tifffile import imwrite
import h5py
import os
from emdfile import tqdmnd
from scipy.ndimage import rotate
from abtem.parametrizations import (
    KirklandParametrization,
    LobatoParametrization,
    PengParametrization,
)
from abtem.core.backend import get_array_module
from abtem.measurements import Images
from abtem.potentials.iam import PotentialArray
import pickle
from scipy.signal import argrelmax

def Process_line_scan(
    measurement,
    label,
    start_point,
    end_point,
    scan_label="1",
    output_folder="Plots/Lineprofile/New_position"
):
    """
    Normalize measurement to 8-bit, convert to abtem Images, and generate a single line scan plot.

    Parameters:
    - measurement: abtem measurement object
    - label: str, e.g., "BF", "MAADF", "HAADF"
    - start_point: tuple(float, float), starting coordinate of the scan
    - end_point: tuple(float, float), ending coordinate of the scan
    - scan_label: str or int, used to differentiate multiple scans (default "1")
    - output_folder: str, directory path to save plots
    """

    os.makedirs(output_folder, exist_ok=True)

    # Normalize and convert to 8-bit
    data = measurement.array.squeeze()
    data_min = data.min()
    data_max = data.max()
    data_normalized = (data - data_min) / (data_max - data_min)
    data_8bit = (data_normalized * 255).astype(np.uint8)

    # Convert to abtem format
    image = Images(
        data_8bit,
        measurement.sampling,
        measurement.ensemble_axes_metadata,
        measurement.metadata,
    )

    # Interpolate line profile
    line_profile = image.interpolate_line(start=start_point, end=end_point, width=2)
    line_profile.show()
    profile_path = os.path.join(output_folder, f"Line_profile_{label}_{scan_label}.png")
    plt.savefig(profile_path, dpi=600, bbox_inches="tight")
    plt.close()

    # Plot image with line overlay
    fig, ax = plt.subplots()
    image.show(ax=ax, explode=True, cbar=True, vmin=image.min(), vmax=image.max())
    ax.scatter(start_point[0], start_point[1], color='red', label='Start')
    ax.scatter(end_point[0], end_point[1], color='blue', label='End')
    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red')
    ax.legend()
    plt.show()
    scan_path = os.path.join(output_folder, f"Linescan_{label}_{scan_label}.png")
    plt.savefig(scan_path, dpi=600, bbox_inches="tight")
    plt.close()
    return line_profile


# ------------------ Find peaks of lineprofiles ------------------ #
def print_peak_values(r_values, bf_values, maadf_values, haadf_values, start_point, sample_type, threshold=1):
    """
    This function finds peaks in the data and prints their corresponding values.
    It uses the `argrelmax` function to find local maxima.

    :param r_values: The r-values (distance values).
    :param bf_values: The BF intensity values.
    :param maadf_values: The MAADF intensity values.
    :param haadf_values: The HAADF intensity values.
    :param start_point: The start point of the line scan.
    :param threshold: A threshold value to consider only significant peaks (optional).
    """

    print(f"\nPeaks for {sample_type} (start={start_point}):")

    bf_peaks = argrelmax(bf_values)[0]
    maadf_peaks = argrelmax(maadf_values)[0]
    haadf_peaks = argrelmax(haadf_values)[0]

    print("\nBF Intensity Peaks:")
    for peak in bf_peaks:
        if bf_values[peak] > threshold:
            print(f"At r = {r_values[peak]:.2f} Å, BF = {bf_values[peak]:.2f} a.u.")

    print("\nMAADF Intensity Peaks:")
    for peak in maadf_peaks:
        if maadf_values[peak] > threshold:
            print(f"At r = {r_values[peak]:.2f} Å, MAADF = {maadf_values[peak]:.2f} a.u.")

    print("\nHAADF Intensity Peaks:")
    for peak in haadf_peaks:
        if haadf_values[peak] > threshold:
            print(f"At r = {r_values[peak]:.2f} Å, HAADF = {haadf_values[peak]:.2f} a.u.")


