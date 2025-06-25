"""
Compute geometric contact angle from segmented image data.

Output is txt file with contact angle values.
"""
from geometricCA import compute_geometric_contact_angle

# Configuration
file_path   = 'MW_fw50-SV.tif'
fluid_Label = 1 # Light phase fluid, in case of brine-gas system (label is for gas phase)
solid_Label = 2

# Run computation
compute_geometric_contact_angle(
    file_path, 
    fluidLabel=fluid_Label, 
    solidLabel=solid_Label
)
