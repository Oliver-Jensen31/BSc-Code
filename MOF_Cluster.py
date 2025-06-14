import abtem
import numpy as np
import ase
import ase.spacegroup
from ase.cluster import wulff_construction
from ase.spacegroup import crystal
from ase.io import read
from abtem.visualize import show_atoms
import matplotlib.pyplot as plt
from tifffile import imwrite
import h5py
from emdfile import tqdmnd
from scipy.ndimage import rotate
from abtem.measurements import Images
from abtem.parametrizations import (
    KirklandParametrization,
    LobatoParametrization,
    PengParametrization,
)
from abtem.core.backend import get_array_module
from abtem.potentials.iam import PotentialArray
import pickle



#---------------------Creating the structure---------------------#
MOF = read('cif/MOF.cif')

show_atoms(MOF,plane="xy", legend=True)
plt.savefig("Plots/MOF_Cluster/MOF_xy.png", dpi=600, bbox_inches="tight")
show_atoms(MOF,plane="xz", legend=True)
plt.savefig("Plots/MOF_Cluster/MOF_xz.png", dpi=600, bbox_inches="tight")
show_atoms(MOF,plane="yz", legend=True)
plt.savefig("Plots/MOF_Cluster/MOF_yz.png", dpi=600, bbox_inches="tight")


#---------------------Creating the cluster---------------------#
cluster = read('cif/Pt9_without_ligands.xyz')

show_atoms(cluster,plane="xy", legend=True)
plt.savefig("Plots/MOF_Cluster/Cluster_xy.png", dpi=600, bbox_inches="tight")
show_atoms(cluster,plane="xz", legend=True)
plt.savefig("Plots/MOF_Cluster/Cluster_xz.png", dpi=600, bbox_inches="tight")
show_atoms(cluster,plane="yz", legend=True)
plt.savefig("Plots/MOF_Cluster/Cluster_yz.png", dpi=600, bbox_inches="tight")

#---------------------Rotate MOF---------------------#
Rotated_structure = MOF.copy()
Rotated_structure.rotate(90, "y")

# Rotate the structure to counteract the tilt of the MOF
c = Rotated_structure.cell[2]
theta_y = np.degrees(np.arctan2(c[0], c[2])) 
Rotated_structure.rotate(-theta_y, 'y', rotate_cell=True)
thickness = 1
Rotated_structure = Rotated_structure * (2,2,thickness)

show_atoms(Rotated_structure,plane="xy", legend=True)
plt.savefig("Plots/MOF_Cluster/Rotated_structure_xy.png", dpi=600, bbox_inches="tight")
show_atoms(Rotated_structure,plane="xz", legend=True)
plt.savefig("Plots/MOF_Cluster/Rotated_structure_xz.png", dpi=600, bbox_inches="tight")
show_atoms(Rotated_structure,plane="yz", legend=True)
plt.savefig("Plots/MOF_Cluster/Rotated_structure_yz.png", dpi=600, bbox_inches="tight")



#---------------------Cluster inside MOF---------------------#
cluster.rotate('y', 90)
cluster.translate((-17.5,0,5))
Combined_structure = cluster + Rotated_structure
Combined_structure.center(vacuum=2)


show_atoms(Combined_structure,plane="xy", legend=True)
plt.savefig("Plots/MOF_Cluster/Combined_structure_xy.png", dpi=600, bbox_inches="tight")
show_atoms(Combined_structure,plane="xz", legend=True)
plt.savefig("Plots/MOF_Cluster/Combined_structure_xz.png", dpi=600, bbox_inches="tight")
show_atoms(Combined_structure,plane="yz", legend=True)
plt.savefig("Plots/MOF_Cluster/Combined_structure_yz.png", dpi=600, bbox_inches="tight")


#---------------------Thickness---------------------#
cell = Combined_structure.get_cell()
c_length = np.linalg.norm(cell[2])
print("Total thickness (Å):", c_length)


#---------------------Potential---------------------#
slice_thickness = 1
potential = abtem.Potential(
    Combined_structure,
    parametrization="lobato",
    slice_thickness=slice_thickness,
    projection="finite",
    sampling=0.05,
)

print("grid  (Nz, Ny, Nx):", potential.shape)


#---------------------Probe---------------------#
probe = abtem.Probe(energy=300e3, 
                    semiangle_cutoff=20,
                    Cs=30e4,
                    defocus="scherzer")
probe.match_grid(potential)

print(f"Maximum simulated scattering angle = {min(probe.cutoff_angles):.1f} mrad")
sampling = probe.aperture.nyquist_sampling

print(f"Nyquist sampling = {sampling:.3f} Å/pixel")
print(f"defocus = {probe.aberrations.defocus} Å")
print(f"FWHM = {probe.profiles().width().compute()} Å")

#---------------------Scanning---------------------#
scan_step_size = 0.1
grid_scan = abtem.GridScan(
    start=(0, 0),
    end=(24, 18),
    sampling=scan_step_size,
    potential=potential,
)

fig_1, ax = abtem.show_atoms(Combined_structure, legend=True)
grid_scan.add_to_plot(ax)
fig_1.savefig("Plots/MOF_Cluster/grid_scan.png", dpi=600, bbox_inches="tight")
plt.close()


#---------------------Detector---------------------#
pixelated_detector = abtem.PixelatedDetector(
    max_angle=200,
)

measurement = probe.scan(
    potential,
    scan=grid_scan,
    detectors=pixelated_detector,
)
measurement.compute()


#---------------------Save measurement---------------------#
with open("/work3/s224054/Cluster.pkl", "wb") as f:
    pickle.dump(measurement, f, protocol=pickle.HIGHEST_PROTOCOL)


#---------------------Imaging---------------------#
bf_measurement = measurement.integrate_radial(0, 25)
maadf_measurement = measurement.integrate_radial(50, 100)
haadf_measurement = measurement.integrate_radial(120, 180)


# Normalize and convert BF to 8-bit
bf_data = bf_measurement.array.squeeze()
bf_min = bf_data.min()
bf_max = bf_data.max()
bf_normalized = (bf_data - bf_min) / (bf_max - bf_min)
bf_8bit = (bf_normalized * 255).astype(np.uint8)

# Normalize and convert MAADF to 8-bit
maadf_data = maadf_measurement.array.squeeze()
maadf_min = maadf_data.min()
maadf_max = maadf_data.max()
maadf_normalized = (maadf_data - maadf_min) / (maadf_max - maadf_min)
maadf_8bit = (maadf_normalized * 255).astype(np.uint8)

# Normalize and convert HAADF to 8-bit
haadf_data = haadf_measurement.array.squeeze()
haadf_min = haadf_data.min()
haadf_max = haadf_data.max()
haadf_normalized = (haadf_data - haadf_min) / (haadf_max - haadf_min)
haadf_8bit = (haadf_normalized * 255).astype(np.uint8)


bf_image = Images(
    bf_8bit,
    bf_measurement.sampling,
    bf_measurement.ensemble_axes_metadata,
    bf_measurement.metadata,
)
maadf_image = Images(
    maadf_8bit,
    maadf_measurement.sampling,
    maadf_measurement.ensemble_axes_metadata,
    maadf_measurement.metadata,
)
haadf_image = Images(
    haadf_8bit,
    haadf_measurement.sampling,
    haadf_measurement.ensemble_axes_metadata,
    haadf_measurement.metadata,
)

# Stack and show all
measurements = abtem.stack(
    [bf_measurement, maadf_measurement, haadf_measurement], ("BF", "MAADF", "HAADF")
)

measurements.show(
    explode=True,
    figsize=(14, 5),
    cbar=True,
)
plt.savefig(f"Plots/MOF_Cluster/BF_ADF_HAADF_x{thickness}_St{slice_thickness}_S{scan_step_size}.png", dpi=600, bbox_inches="tight")



# Show and save individual channels
bf_image.show(explode=False, figsize=(14, 5), cbar=True)
plt.savefig(f"Plots/MOF_Cluster/BF_x{thickness}_St{slice_thickness}_S{scan_step_size}.png", dpi=600, bbox_inches="tight")

maadf_image.show(explode=False, figsize=(14, 5), cbar=True)
plt.savefig(f"Plots/MOF_Cluster/MAADF_x{thickness}_St{slice_thickness}_S{scan_step_size}.png", dpi=600, bbox_inches="tight")

haadf_image.show(explode=False, figsize=(14, 5), cbar=True)
plt.savefig(f"Plots/MOF_Cluster/HAADF_x{thickness}_St{slice_thickness}_S{scan_step_size}.png", dpi=600, bbox_inches="tight")





