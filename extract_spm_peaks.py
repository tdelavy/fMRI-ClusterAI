import re
def extract_spm_clusters(in_file, out_file):
  
    with open(in_file, 'r') as f:
        content = f.read()

    # Compile the regexes as before.
    cluster_size_pattern = re.compile(
        r"text\([^)]*Parent',axes6[^)]*FontWeight','bold'[^)]*String','(\d+)'[^)]*Position',\[\s*0\.33\s*[^)]*\]",
        re.DOTALL
    )
    peak_coord_pattern = re.compile(
        r"text\([^)]*Tag','ListXYZ'[^)]*Parent',axes6[^)]*FontWeight','bold'[^)]*String','([^']+)'",
        re.DOTALL
    )

    clusters = []
    # Create iterators for both patterns.
    size_iter = cluster_size_pattern.finditer(content)
    coord_iter = peak_coord_pattern.finditer(content)
    
    # Convert the iterators to lists to process them sequentially.
    size_results = list(size_iter)
    coord_results = list(coord_iter)
    
    # Assuming each cluster size corresponds to one peak coordinate in order.
    for size_match, coord_match in zip(size_results, coord_results):
        try:
            cluster_size = int(size_match.group(1))
        except ValueError:
            continue
        coord_str = coord_match.group(1).strip()
        parts = coord_str.split()
        if len(parts) == 3:
            try:
                x, y, z = map(float, parts)
                clusters.append((cluster_size, x, y, z))
            except ValueError:
                pass

    # Write the extracted data to a .1D file.
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
