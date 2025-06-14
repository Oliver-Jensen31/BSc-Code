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
from emdfile import tqdmnd
from numpy.linalg import norm
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
from Functions import *

#------------------Load the data------------------#
with open("/work3/s224054/Cluster.pkl", "rb") as f:
    measurement_2 = pickle.load(f)
bf_measurement_2 = measurement_2.integrate_radial(0, 25)
maadf_measurement_2 = measurement_2.integrate_radial(50, 100)
haadf_measurement_2 = measurement_2.integrate_radial(120, 180)

#------------------Set points------------------#
start_point1 = (0.5, 2)
end_point1 = (14.5, 2)
start_point2 = (8, 8)
end_point2 = (20, 8)
start_point3 = (11, 0.5)
end_point3 = (18.5, 11)

# ------------------ Line Profile Extraction ------------------ #
# Cluster 1
bf_cluster_1    = Process_line_scan(bf_measurement_2, "BF_Cluster",    start_point=start_point1, end_point=end_point1, scan_label="1")
maadf_cluster_1 = Process_line_scan(maadf_measurement_2, "MAADF_Cluster", start_point=start_point1, end_point=end_point1, scan_label="1")
haadf_cluster_1 = Process_line_scan(haadf_measurement_2, "HAADF_Cluster", start_point=start_point1, end_point=end_point1, scan_label="1")

n_points = bf_cluster_1.array.shape[-1]
physical_distance = norm(np.array(end_point1) - np.array(start_point1))
r_values = np.linspace(0, physical_distance, n_points)

bf_inv = bf_cluster_1.array.squeeze()
bf_inv = bf_inv.max() - bf_inv  

plt.figure()
plt.plot(r_values, bf_inv, label="BF (inverted)")
plt.plot(r_values, maadf_cluster_1.array.squeeze(), label="MAADF")
plt.plot(r_values, haadf_cluster_1.array.squeeze(), label="HAADF")
plt.xlabel("r [Å]")
plt.ylabel("Intensity (a.u.)")
plt.legend()
plt.tight_layout()
plt.savefig("Plots/Lineprofile/New_position/Combined_Line_Profile_Cluster1.png", dpi=600)
plt.close()

# Cluster 2
bf_cluster_2    = Process_line_scan(bf_measurement_2, "BF_Cluster",    start_point=start_point2, end_point=end_point2, scan_label="2")
maadf_cluster_2 = Process_line_scan(maadf_measurement_2, "MAADF_Cluster", start_point=start_point2, end_point=end_point2, scan_label="2")
haadf_cluster_2 = Process_line_scan(haadf_measurement_2, "HAADF_Cluster", start_point=start_point2, end_point=end_point2, scan_label="2")

n_points = bf_cluster_2.array.shape[-1]
physical_distance = norm(np.array(end_point2) - np.array(start_point2))
r_values = np.linspace(0, physical_distance, n_points)

bf_inv = bf_cluster_2.array.squeeze()
bf_inv = bf_inv.max() - bf_inv

plt.figure()
plt.plot(r_values, bf_inv, label="BF (inverted)")
plt.plot(r_values, maadf_cluster_2.array.squeeze(), label="MAADF")
plt.plot(r_values, haadf_cluster_2.array.squeeze(), label="HAADF")
plt.xlabel("r [Å]")
plt.ylabel("Intensity (a.u.)")
plt.legend()
plt.tight_layout()
plt.savefig("Plots/Lineprofile/New_position/Combined_Line_Profile_Cluster2.png", dpi=600)
plt.close()

# Cluster 3
bf_cluster_3    = Process_line_scan(bf_measurement_2, "BF_Cluster",    start_point=start_point3, end_point=end_point3, scan_label="3")
maadf_cluster_3 = Process_line_scan(maadf_measurement_2, "MAADF_Cluster", start_point=start_point3, end_point=end_point3, scan_label="3")
haadf_cluster_3 = Process_line_scan(haadf_measurement_2, "HAADF_Cluster", start_point=start_point3, end_point=end_point3, scan_label="3")

n_points = bf_cluster_3.array.shape[-1]
physical_distance = norm(np.array(end_point3) - np.array(start_point3))
r_values = np.linspace(0, physical_distance, n_points)

bf_inv = bf_cluster_3.array.squeeze()
bf_inv = bf_inv.max() - bf_inv

plt.figure()
plt.plot(r_values, bf_inv,                         label="BF (inverted)")
plt.plot(r_values, maadf_cluster_3.array.squeeze(), label="MAADF")
plt.plot(r_values, haadf_cluster_3.array.squeeze(), label="HAADF")
plt.xlabel("r [Å]")
plt.ylabel("Intensity (a.u.)")
plt.legend()
plt.tight_layout()
plt.savefig("Plots/Lineprofile/New_position/Combined_Line_Profile_Cluster3.png", dpi=600)
plt.close()

# Print peak values
bf_values = bf_cluster_3.array.squeeze()
bf_values = bf_values.max() - bf_values
maadf_values = maadf_cluster_3.array.squeeze()
haadf_values = haadf_cluster_3.array.squeeze()
print_peak_values(r_values, bf_values, maadf_values, haadf_values, start_point=start_point3, sample_type="Cluster")

