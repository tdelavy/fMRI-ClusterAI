import re

def extract_spm_clusters(in_file, out_file):
    """
    Reads a MATLAB script (e.g., ExempleSubj1.m) and extracts the bold cluster voxel
    count and the bold peak coordinate (assumed to be the main peak for that cluster)
    and writes them to a .1D file.

    The .1D file will have the header:
        #Coordinate order = LPI (for SPM)
        #Voxels  Peak x  Peak y  Peak z
        #------  ------  ------  ------
    And each subsequent line will be formatted as:
         vox   +x.x   +y.y   +z.z

    This script assumes that the cluster voxel count is on a bold text line (using 
    FontWeight 'bold') and that the main peak coordinate is on a bold text line with Tag 'ListXYZ'.
    """
    # Pattern for cluster voxel count (bold)
    cluster_size_pattern = re.compile(
        r"text\([^)]*FontWeight','bold','String','(\d+)'", re.DOTALL
    )
    # Pattern for bold peak coordinate (with Tag 'ListXYZ')
    peak_coord_pattern = re.compile(
        r"text\([^)]*Tag','ListXYZ','FontWeight','bold','String','([^']+)'", re.DOTALL
    )

    clusters = []  # List of tuples: (cluster_size, x, y, z)
    current_cluster_size = None

    with open(in_file, 'r') as f:
        # Read the entire file to allow matching multi-line constructs if needed
        content = f.read()
        # Process line by line
        for line in content.splitlines():
            line = line.strip()
            # Check for a cluster size line first
            size_match = cluster_size_pattern.search(line)
            if size_match:
                try:
                    current_cluster_size = int(size_match.group(1))
                except ValueError:
                    current_cluster_size = None
                continue

            # Check for a bold peak coordinate line (with Tag 'ListXYZ')
            peak_match = peak_coord_pattern.search(line)
            if peak_match and current_cluster_size is not None:
                coord_str = peak_match.group(1).strip()
                parts = coord_str.split()
                if len(parts) == 3:
                    try:
                        x, y, z = map(float, parts)
                        clusters.append((current_cluster_size, x, y, z))
                    except ValueError:
                        pass
                # Reset current_cluster_size so that subsequent peaks (if any) for the same cluster are not duplicated.
                current_cluster_size = None

    # Write the extracted data to a .1D file with the requested header and formatting.
    with open(out_file, 'w') as f:
        f.write("#Coordinate order = LPI (SPM)\n")
        f.write("#Voxels  Peak x  Peak y  Peak z\n")
        f.write("#------  ------  ------  ------\n")
        for (vox, x, y, z) in clusters:
            f.write(f"{vox:>6d}  {x:>7.1f}  {y:>7.1f}  {z:>7.1f}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python extract_spm_clusters.py <in_file.m> <out_file.1D>")
        sys.exit(1)
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    extract_spm_clusters(in_file, out_file)
    print(f"Cluster peaks written to {out_file}")
