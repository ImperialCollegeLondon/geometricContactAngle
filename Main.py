from ComputeGCA import compute_geometric_contact_angle
file_path = 'MW_fw50-SV.tif'
fluidLabel = 1
SolidLabel = 3

compute_geometric_contact_angle(file_path, fluidLabel=fluidLabel, solidLabel=SolidLabel, min_points=20)