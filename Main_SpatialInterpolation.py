"""
Spatially interpolate contact angles from geometric measurements.
This script takes the output from the geometric contact angle measurement
and creates a 3D spatial distribution of contact angles throughout the
pore space.

Output includes:
- 3D contact angle field (TIFF)
- Statistical analysis and visualizations
"""
from SpatialInterpolation import ContactAngleInterpolator

# Configuration
image_path = 'MW_fw02SV.tif'  # Segmented image
ca_data_path = 'MW_fw50-SV_CAdata.txt'  # Contact angle measurements from Main.py
primary_drainage = True  # True, if this image was taken after the primary drainage process and before water injection (fw=0).

solid_label = 2
fluid1_label = 0 # Index label for the denser phase (e.g., brine)
fluid2_label = 1 # Index label for the lighter phase (e.g., oil or gas)


#----------------------------------------------------------------Run computation------------------------------------------------------------------------------------
interpolator = ContactAngleInterpolator(image_path, ca_data_path, solid_label, fluid2_label, fluid1_label, primary_drainage)
interpolator.parse_contact_angle_file(use_line_means=True)
interpolator.map_coordinates_to_voxels(scale_to_image=True)
interpolator.interpolate_angles(
    oil_method='weighted_mean',
    brine_method='idw',
    interpolation_threshold=3,
    max_distance=20
)
interpolator.visualize_results(save_path='contact_angle_analysis.png')
interpolator.save_results('contact_angle_distribution.tif', save_metadata=True)