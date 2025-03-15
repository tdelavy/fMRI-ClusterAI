import streamlit as st
import subprocess
import tempfile
import os
from openai import OpenAI
import re
import docx
from docx.shared import Pt
import nibabel as nib
import numpy as np

st.set_page_config(page_title="AItlas Clusters", page_icon="🧠")

# Instantiate the client with the Perplexity API endpoint.
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set your OPENAI_API_KEY environment variable!")
else:
    client = OpenAI(api_key=openai_api_key, base_url="https://api.perplexity.ai")

def load_label_file(file_path):
    """
    Reads a label file (in the format produced by AFNI's 3dinfo -labeltable)
    and returns a dictionary mapping integer labels to region names.
    
    Expected file format (each line):
      "9" "right_Area_hOc4v_(LingG)"
      "10" "right_Area_hOc3v_(LingG)"
      ...
    """
    label_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith('"'):
                continue
            matches = re.findall(r'"(.*?)"', line)
            if len(matches) >= 2:
                try:
                    key = int(matches[0])
                    label_dict[key] = matches[1]
                except ValueError:
                    continue
    return label_dict

# Load label dictionaries from your text files.
label_julich = load_label_file("labelJulich.txt")
label_fs_afni = load_label_file("labelFs.afni.txt")

def search_all_labels(voxel, atlas_data, max_radius=7):
    """
    Searches around the given voxel (within a sphere of radius max_radius)
    for all nonzero labels and returns a dictionary mapping label values
    to their minimum Euclidean distance (in voxel units) from the original voxel.
    """
    shape = atlas_data.shape
    label_distances = {}
    for di in range(-max_radius, max_radius+1):
        for dj in range(-max_radius, max_radius+1):
            for dk in range(-max_radius, max_radius+1):
                # Only consider offsets within a sphere of radius max_radius.
                if di**2 + dj**2 + dk**2 <= max_radius**2:
                    ni = voxel[0] + di
                    nj = voxel[1] + dj
                    nk = voxel[2] + dk
                    if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
                        lab = int(atlas_data[ni, nj, nk])
                        if lab != 0:
                            dist = (di**2 + dj**2 + dk**2)**0.5
                            if lab in label_distances:
                                label_distances[lab] = min(label_distances[lab], dist)
                            else:
                                label_distances[lab] = dist
    return label_distances

def get_anatomical_labels(coord, atlas, coord_system="RAI"):
    """
    Maps a coordinate (x, y, z) to the correct voxel index using AFNI's 
    geometry string: i = 96 - x,  j = 132 - y,  k = z + 78.
    
    Includes debug print statements to display intermediate values.
    If the direct lookup returns 0, it searches within a 7 mm radius for nearby nonzero labels.
    Then uses the label dictionary to return the region name(s) with distance information.
    """
    # 1) Choose which atlas to load
    if atlas == "Julich_MNI2009c":
        atlas_path = "Julich_MNI2009c.nii.gz"
        label_dict = label_julich
    elif atlas == "FS.afni.MNI2009c_asym":
        atlas_path = "FS.afni.MNI2009c_asym.nii.gz"
        label_dict = label_fs_afni
    else:
        return "Unknown atlas"
    
    # 2) Load the atlas data (ignoring nibabel’s affine for coordinate transform)
    try:
        atlas_img = nib.load(atlas_path)
    except Exception as e:
        return f"Error loading atlas: {e}"
    atlas_data = atlas_img.get_fdata()
    shape = atlas_data.shape  # e.g. (193, 229, 193)
    
    #st.write("Atlas shape:", shape)
    
    # 3) Convert the input coordinate from the user’s chosen system to RAS.
    #    Here we assume your cluster file’s coordinates are already effectively in RAS.
    if coord_system.upper() == "RAI":
        conv_coord = (coord[0], coord[1], coord[2])
    elif coord_system.upper().startswith("LPI"):
        conv_coord = (-coord[0], -coord[1], coord[2])
    else:
        conv_coord = coord
    
    #st.write("Original coordinate:", coord)
    #st.write("Converted coordinate (assumed RAS):", conv_coord)
    
    x_ras, y_ras, z_ras = conv_coord
    
    # 4) Apply AFNI's geometry string formula: i = 96 - x, j = 132 - y, k = z + 78.
    i = 96 - x_ras
    j = 132 - y_ras
    k = z_ras + 78
    
    # 5) Round to nearest integer for voxel indices.
    i, j, k = int(round(i)), int(round(j)), int(round(k))
    
    #st.write("Computed voxel indices: i=", i, " j=", j, " k=", k)
    
    # 6) Check if voxel is in bounds.
    if i < 0 or i >= shape[0] or j < 0 or j >= shape[1] or k < 0 or k >= shape[2]:
        return "Coordinate outside atlas volume"
    
    # 7) Direct lookup.
    label_val = int(atlas_data[i, j, k])
    #st.write("Direct lookup atlas label value at voxel:", label_val)
    
    if label_val != 0:
        region_name = label_dict.get(label_val, f"Label {label_val} not found")
        return region_name
    else:
        # 8) If direct lookup returns 0, search within a 7 mm radius for nearby nonzero labels.
        #st.write("Direct lookup returned 0. Searching for nearby labels within 7 mm...")
        label_candidates = search_all_labels((i, j, k), atlas_data, max_radius=7)
        if label_candidates:
            # Sort candidate labels by distance.
            sorted_candidates = sorted(label_candidates.items(), key=lambda x: x[1])
            search_info = "\n".join([f"* Within {round(dist,1)} mm: {label_dict.get(lab, 'Label '+str(lab))}" for lab, dist in sorted_candidates])
            return search_info
        else:
            return "Label 0 not found"

def parse_cluster_file(file_path, max_clusters=6):
    clusters = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 7:
                # AFNI format: [vox, CM_x, CM_y, CM_z, Peak_x, Peak_y, Peak_z, ...]
                cluster = {
                    'voxels': int(parts[0]),
                    'CM': (float(parts[1]), float(parts[2]), float(parts[3])),
                    'Peak': (float(parts[4]), float(parts[5]), float(parts[6]))
                }
            elif len(parts) == 4:
                # SPM conversion format: [vox, Peak_x, Peak_y, Peak_z]
                cluster = {
                    'voxels': int(parts[0]),
                    'CM': None,
                    'Peak': (float(parts[1]), float(parts[2]), float(parts[3]))
                }
            else:
                continue
            clusters.append(cluster)
            if len(clusters) == max_clusters:  # Only keep the first 6 clusters.
                break
    return clusters
    
def synthesize_interpretation(anatomical_info, task_description, contrast_description, max_clusters):
    """
    Sends a prompt to the Perplexity API, streams tokens, and progressively
    updates Streamlit to show partial output as it arrives.
    """
    prompt = (
        "You are a knowledgeable yet objective research assistant specialized in neuroscience. "
        f"You will analyze the user-provided anatomical information for up to {max_clusters} clusters, as determined by AFNI whereami, in the context of an fMRI study.\n"
        "Please follow these steps and rules:\n"
        "1) Review each cluster.\n"
        "2) Explain the cluster’s involvement in the fMRI task (if any relevant evidence or prior studies exist). If there is 0 known involvement, say so.\n"
        "3) Summarize any up-to-date literature from reputable sources, if available, on that region’s role in the tasks or processes. Make sure you give enough relevant information to understand the role of the cluster in known litterature. But, if no relevant studies are found, say that evidence is currently limited or inconclusive.\n"
        "\n"
        f"Task: {task_description}\n"
        f"Contrast: {contrast_description}\n"
        "Below is the full anatomical labels output (via AFNI whereami) for the relevant clusters:\n"
        f"{anatomical_info}\n\n"
        "Now synthesize a clear, accurate interpretation for each clusters."
    )
    
    # Make a SINGLE call with model="sonar-pro" or "sonar"
    response = client.chat.completions.create(
        model="sonar-deep-research", #-deep-research
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

        # Extract the raw text
    interpretation = response.choices[0].message.content
    
    # Strip out any <think>...</think> segments, if present
    interpretation = re.sub(r"<think>.*?</think>", "", interpretation, flags=re.DOTALL)

    references = getattr(response, "citations", [])

    # Then extract the text
    return interpretation, references

def filter_label_info(output):
    """
    Filters the whereami output to only include lines that begin with "Atlas " or "* Within",
    or that mention "not near any region".
    """
    filtered_lines = []
    for line in output.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith("Atlas ") or line_stripped.startswith("* Within") or "not near any region" in line_stripped:
            filtered_lines.append(line_stripped)
    return "\n".join(filtered_lines)

def run_analysis(cluster_file_path, atlas, task_description, contrast_description, use_ai):
    """
    Runs the full analysis:
      - Parses the cluster file (keeping only the first 6 clusters).
      - Obtains anatomical labels for each cluster.
      - Combines the outputs into a string.
      - Calls ChatGPT for interpretation.
    Returns both the raw anatomical outputs and ChatGPT's interpretation.
    """
    clusters = parse_cluster_file(cluster_file_path, max_clusters)
    anatomical_results = []
    
    # Display the first 6 clusters.
    st.subheader(f"First {max_clusters} clusters:")
    
    # Get anatomical labels for each cluster.
    for i, cluster in enumerate(clusters, start=1):
        label_info = get_anatomical_labels(cluster["Peak"], atlas, coord_system)
        st.write(f"**Cluster {i}:** Voxels: {cluster['voxels']}, Peak: {cluster['Peak']}")
        st.text(label_info)
        st.write("-" * 20)
        anatomical_results.append(f"Cluster {i} (Voxels: {cluster['voxels']}, Peak: {cluster['Peak']}):\n{label_info}")
        
    anatomical_str = "\n\n".join(anatomical_results)

    # ---- Show a spinner while the model is generating the interpretation ----
    with st.spinner("Perplexity is thoroughly researching the internet to find the implications of these identified brain regions for the task and contrast. Please check back in a few minutes! :)"):
        if use_ai:
            interpretation, references = synthesize_interpretation(anatomical_str, task_description, contrast_description, max_clusters)
        else:
            interpretation, references = "AI interpretation disabled", []
        
    return anatomical_str, interpretation, references

def sanitize_filename(s):
    # Replace non-word characters with underscores
    return re.sub(r'\W+', '_', s)

def create_word_report(settings_summary, anatomical_str, interpretation, references):
    """
    Creates a Word document containing the settings summary,
    the parsed clusters, and the ChatGPT interpretation.
    Returns the path to the temporary .docx file.
    """
    doc = docx.Document()

    # Title
    title = doc.add_heading('fMRI Cluster Analysis Report', 0)
    title.alignment = 1  # center

    # Analysis settings section
    doc.add_heading('Analysis Settings', level=1)
    para = doc.add_paragraph(settings_summary)
    para.style.font.size = Pt(11)

    # Parsed Clusters section
    doc.add_heading('Parsed Clusters', level=1)
    doc.add_paragraph(anatomical_str, style='List Bullet')

    # ChatGPT Interpretation section
    doc.add_heading('Perplexity Interpretation', level=1)
    doc.add_paragraph(interpretation, style='Normal')

    # References section
    if references:
        doc.add_heading('References', level=1)
        for i, ref in enumerate(references, start=1):
            doc.add_paragraph(f"[{i}] {ref}", style='List Number')
    else:
        doc.add_heading('References', level=1)
        doc.add_paragraph("No references returned.")

    # Save to a temporary file
    temp_doc = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc.save(temp_doc.name)
    return temp_doc.name

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title("fMRI Cluster Analysis with Atlas Labeling and Perplexity AI")

st.markdown("""
**Description:**  
This app reads an AFNI cluster table file, using the default atlas **FS.afni.MNI2009c_asym**,
to identify anatomical regions for the clusters (voxels). 

The extracted anatomical information is then analyzed by Perplexity’s Sonar Deep Research, which returns relevant literature on the first six clusters for the specified task and condition.
""")

# Sidebar inputs
st.sidebar.header("Analysis Settings")

cluster_file_path = None  # This will store the path to the .1D file for analysis

# Sidebar: Choose software first
conversion_choice = st.sidebar.selectbox(
    label="Software used for clusterising",
    options=["-- Select an option --", "AFNI", "SPM"],
    index=0,
    help="Select 'AFNI' if you already have a .1D cluster file; select 'SPM' if you have an SPM .m file that needs conversion."
)

# Check if a valid option has been selected
if conversion_choice == "-- Select an option --":
    st.warning("AItlas works by reading a .1D file that contains the voxel numbers along with the X, Y, and Z coordinates of the peak of each cluster. It uses these values to identify the anatomical regions for each cluster by referencing the selected atlas, ultimately generating a detailed analysis report.")
    st.write("Please select either AFNI or SPM to continue.")
    st.stop()

if conversion_choice == "SPM":
    task_description = st.sidebar.text_input(
        "Task Description",
        "Stroop task",
        disabled=False,
        help="Locked when SPM is selected."
    )
    contrast_description = st.sidebar.text_input(
        "Contrast Description",
        "",
        disabled=False,
        help="Enter your contrast (e.g., Incongruent minus Congruent)"
    )

if conversion_choice == "SPM":
    # Uploader for SPM .m file only
    uploaded_m_file = st.sidebar.file_uploader(
        "Choose your SPM .m file (MNI152_2009_template as the reference) for conversion",
        type=["m"]
    )
    if uploaded_m_file is not None:
        # Write the uploaded .m file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m") as tmp_m:
            tmp_m.write(uploaded_m_file.read())
            tmp_m_name = tmp_m.name
        # Construct a meaningful output filename using the task and contrast (assumed already defined)
        output_filename = f"SPM_Clusters_{sanitize_filename(task_description)}_{sanitize_filename(contrast_description)}.1D"
        # Use a temporary file for the conversion output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".1D") as tmp_out:
            output_filepath = tmp_out.name
        # Call the external conversion script (extract_spm_peaks.py)
        result = subprocess.run(
            ["python", "extract_spm_peaks.py", tmp_m_name, output_filepath],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Read the content of the converted file
        with open(output_filepath, "r") as f:
            file_content = f.read()

        st.session_state.converted_file_content = file_content

        # Clean up temporary files
        os.remove(tmp_m_name)
        os.remove(output_filepath)

else:
    task_description = st.sidebar.text_input(
        "Task Description",
        "Stroop task",
        help="Enter your fMRI task name"
    )
    contrast_description = st.sidebar.text_input(
        "Contrast Description",
        "",
        help="Enter your contrast (e.g., Incongruent minus Congruent)"
    )
    # AFNI is selected: uploader for .1D file only
    uploaded_file = st.sidebar.file_uploader("Choose your cluster file (.1D)", type=["1D"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".1D") as tmp:
            tmp.write(uploaded_file.read())
            cluster_file_path = tmp.name
    
    coord_system = st.sidebar.selectbox(
        label="Coordinate System",
        options=["RAI", "LPI (=SPM)"],
        index=0,
        help="RAI format is used by AFNI by default, while LPI is used by SPM."
    )
    atlas = st.sidebar.selectbox(
        label="Choose your atlas",
        options=["Julich_MNI2009c", "FS.afni.MNI2009c_asym"],
        index=1,
        help="Select which atlas to use for anatomical labeling"
    )
    max_clusters = st.sidebar.number_input(
        "Max. number of clusters to include",
        min_value=1,
        max_value=12,
        value=6,
        step=1,
        help="Select how many clusters will be used in the analysis. Maximum 12 clusters."
    )
    temp_use_ai = st.sidebar.checkbox(
        "Use AI Interpretation",
        value=False,
        help="If unchecked, the app will not send a request to Perplexity and task/contrast fields will be disabled."
    )

    use_ai = False  # Default state

    if temp_use_ai:
        # When the user checks the box, prompt for a password.
        entered_password = st.sidebar.text_input("Enter API password", type="password")
        if entered_password:
            if entered_password == st.secrets["api_password"]:
                use_ai = True
                st.sidebar.success("Password correct!")
            else:
                st.sidebar.error("Incorrect password. AI Interpretation disabled.")
        else:
            st.sidebar.info("Please enter the password to enable AI Interpretation.")

# --- Run Analysis button (only for AFNI option) ---
if conversion_choice == "AFNI":
    if atlas == "Julich_MNI2009c":
        st.image("Julich_MNI2009c_Atlas.png", caption="Julich MNI2009c Atlas")
        with st.expander("View Julich Atlas Region List"):
            st.text("""\

    Atlas Julich_MNI2009c,      314 regions
    ----------- Begin regions for Julich_MNI2009c atlas-----------
    u:right_Area_45_(IFG):1  
    u:left_Area_45_(IFG):201
    u:right_Area_44_(IFG):2  
    u:left_Area_44_(IFG):202
    u:right_Area_Fo1_(OFC):3  
    u:left_Area_Fo1_(OFC):203
    u:right_Area_Fo2_(OFC):4  
    u:left_Area_Fo2_(OFC):204
    u:right_Area_Fo3_(OFC):5  
    u:left_Area_Fo3_(OFC):205
    u:right_Area_hOc5_(LOC):6  
    u:left_Area_hOc5_(LOC):206
    u:right_Area_hOc2_(V2,_18):7  
    u:left_Area_hOc2_(V2,_18):207
    u:right_Area_hOc1_(V1,_17,_CalcS):8  
    u:left_Area_hOc1_(V1,_17,_CalcS):208
    u:right_Area_hOc4v_(LingG):9  
    u:left_Area_hOc4v_(LingG):209
    u:right_Area_hOc3v_(LingG):10 
    u:left_Area_hOc3v_(LingG):210
    u:right_Area_TE_1.0_(HESCHL):11 
    u:left_Area_TE_1.0_(HESCHL):211
    u:right_Area_TE_2.1_(STG):12 
    u:left_Area_TE_2.1_(STG):212
    u:right_Area_TPJ_(STG/SMG):13 
    u:left_Area_TPJ_(STG/SMG):213
    u:right_Area_TE_1.2_(HESCHL):14 
    u:left_Area_TE_1.2_(HESCHL):214
    u:right_Area_TE_3_(STG):15 
    u:left_Area_TE_3_(STG):215
    u:right_Area_TE_1.1_(HESCHL):16 
    u:left_Area_TE_1.1_(HESCHL):216
    u:right_Area_TE_2.2_(STG):17 
    u:left_Area_TE_2.2_(STG):217
    u:right_Area_33_(ACC):18 
    u:left_Area_33_(ACC):218
    u:right_Area_s32_(sACC):19 
    u:left_Area_s32_(sACC):219
    u:right_Area_p32_(pACC):20 
    u:left_Area_p32_(pACC):220
    u:right_Area_Id2_(Insula):21 
    u:left_Area_Id2_(Insula):221
    u:right_Area_Id3_(Insula):22 
    u:left_Area_Id3_(Insula):222
    u:right_CA2_(Hippocampus):23 
    u:left_CA2_(Hippocampus):223
    u:right_CA3_(Hippocampus):24 
    u:left_CA3_(Hippocampus):224
    u:right_Entorhinal_Cortex:25 
    u:left_Entorhinal_Cortex:225
    u:right_DG_(Hippocampus):26 
    u:left_DG_(Hippocampus):226
    u:right_CA1_(Hippocampus):27 
    u:left_CA1_(Hippocampus):227
    u:right_HATA_(Hippocampus):28 
    u:left_HATA_(Hippocampus):228
    u:right_Area_OP4_(POperc):29 
    u:left_Area_OP4_(POperc):229
    u:right_Area_OP1_(POperc):30 
    u:left_Area_OP1_(POperc):230
    u:right_Area_OP2_(POperc):31 
    u:left_Area_OP2_(POperc):231
    u:right_Area_OP3_(POperc):32 
    u:left_Area_OP3_(POperc):232
    u:right_Area_FG2_(FusG):33 
    u:left_Area_FG2_(FusG):233
    u:right_Area_FG1_(FusG):34 
    u:left_Area_FG1_(FusG):234
    u:right_Area_PGp_(IPL):35 
    u:left_Area_PGp_(IPL):235
    u:right_Area_PFt_(IPL):36 
    u:left_Area_PFt_(IPL):236
    u:right_Area_PGa_(IPL):37 
    u:left_Area_PGa_(IPL):237
    u:right_Area_PFop_(IPL):38 
    u:left_Area_PFop_(IPL):238
    u:right_Area_PFm_(IPL):39 
    u:left_Area_PFm_(IPL):239
    u:right_Area_PFcm_(IPL):40 
    u:left_Area_PFcm_(IPL):240
    u:right_Area_Ig2_(Insula):41 
    u:left_Area_Ig2_(Insula):241
    u:right_Area_Ig1_(Insula):42 
    u:left_Area_Ig1_(Insula):242
    u:right_Area_Id1_(Insula):43 
    u:left_Area_Id1_(Insula):243
    u:right_Area_hOc4lp_(LOC):44 
    u:left_Area_hOc4lp_(LOC):244
    u:right_Area_hOc4la_(LOC):45 
    u:left_Area_hOc4la_(LOC):245
    u:right_Area_hOc4d_(Cuneus):46 
    u:left_Area_hOc4d_(Cuneus):246
    u:right_Area_hOc3d_(Cuneus):47 
    u:left_Area_hOc3d_(Cuneus):247
    u:right_Area_4p_(PreCG):48 
    u:left_Area_4p_(PreCG):248
    u:right_Area_4a_(PreCG):49 
    u:left_Area_4a_(PreCG):249
    u:right_Area_1_(PostCG):50 
    u:left_Area_1_(PostCG):250
    u:right_Area_3a_(PostCG):51 
    u:left_Area_3a_(PostCG):251
    u:right_Area_3b_(PostCG):52 
    u:left_Area_3b_(PostCG):252
    u:right_Area_hIP1_(IPS):53 
    u:left_Area_hIP1_(IPS):253
    u:right_Area_hIP2_(IPS):54 
    u:left_Area_hIP2_(IPS):254
    u:right_Area_5L_(SPL):55 
    u:left_Area_5L_(SPL):255
    u:right_Area_5M_(SPL):56 
    u:left_Area_5M_(SPL):256
    u:right_Area_7PC_(SPL):57 
    u:left_Area_7PC_(SPL):257
    u:right_Area_hIP3_(IPS):58 
    u:left_Area_hIP3_(IPS):258
    u:right_Area_7A_(SPL):59 
    u:left_Area_7A_(SPL):259
    u:right_Area_7M_(SPL):60 
    u:left_Area_7M_(SPL):260
    u:right_Area_5Ci_(SPL):61 
    u:left_Area_5Ci_(SPL):261
    u:right_Area_Id7_(Insula):62 
    u:left_Area_Id7_(Insula):262
    u:right_Area_s24_(sACC):63 
    u:left_Area_s24_(sACC):263
    u:right_Area_25_(sACC):64 
    u:left_Area_25_(sACC):264
    u:right_SF_(Amygdala):65 
    u:left_SF_(Amygdala):265
    u:right_LB_(Amygdala):66 
    u:left_LB_(Amygdala):266
    u:right_Subiculum_(Hippocampus):67 
    u:left_Subiculum_(Hippocampus):267
    u:right_Area_PF_(IPL):68 
    u:left_Area_PF_(IPL):268
    u:right_Area_7P_(SPL):69 
    u:left_Area_7P_(SPL):269
    u:right_Area_Fp2_(FPole):70 
    u:left_Area_Fp2_(FPole):270
    u:right_Area_Fp1_(FPole):71 
    u:left_Area_Fp1_(FPole):271
    u:right_Fastigial_Nucleus_(Cerebellum):72 
    u:left_Fastigial_Nucleus_(Cerebellum):272
    u:right_VTM_(Amygdala):73 
    u:left_VTM_(Amygdala):273
    u:right_Area_p24ab_(pACC):74 
    u:left_Area_p24ab_(pACC):274
    u:right_Area_p24c_(pACC):75 
    u:left_Area_p24c_(pACC):275
    u:right_MF_(Amygdala):76 
    u:left_MF_(Amygdala):276
    u:right_IF_(Amygdala):77 
    u:left_IF_(Amygdala):277
    u:right_Area_FG4_(FusG):78 
    u:left_Area_FG4_(FusG):278
    u:right_Area_FG3_(FusG):79 
    u:left_Area_FG3_(FusG):279
    u:right_Dorsal_Dentate_Nucleus_(Cerebellum):80 
    u:left_Dorsal_Dentate_Nucleus_(Cerebellum):280
    u:right_Ventral_Dentate_Nucleus_(Cerebellum):81 
    u:left_Ventral_Dentate_Nucleus_(Cerebellum):281
    u:right_Area_TeI_(STG):82 
    u:left_Area_TeI_(STG):282
    u:right_Area_TI_(STG):83 
    u:left_Area_TI_(STG):283
    u:right_Area_IFS1_(IFS):84 
    u:left_Area_IFS1_(IFS):284
    u:right_Area_IFS2_(IFS):85 
    u:left_Area_IFS2_(IFS):285
    u:right_Area_IFS3_(IFS):86 
    u:left_Area_IFS3_(IFS):286
    u:right_Area_IFS4_(IFS):87 
    u:left_Area_IFS4_(IFS):287
    u:right_Area_IFJ1_(IFS,PreCS):88 
    u:left_Area_IFJ1_(IFS,PreCS):288
    u:right_Area_IFJ2_(IFS,PreCS):89 
    u:left_Area_IFJ2_(IFS,PreCS):289
    u:right_Interposed_Nucleus_(Cerebellum):90 
    u:left_Interposed_Nucleus_(Cerebellum):290
    u:right_Area_2_(PostCS):91 
    u:left_Area_2_(PostCS):291
    u:right_Ch_4_(Basal_Forebrain):92 
    u:left_Ch_4_(Basal_Forebrain):292
    u:right_Area_STS1_(STS):93 
    u:left_Area_STS1_(STS):293
    u:right_Area_STS2_(STS):94 
    u:left_Area_STS2_(STS):294
    u:right_Area_Op8_(Frontal_Operculum):95 
    u:left_Area_Op8_(Frontal_Operculum):295
    u:right_Area_Op9_(Frontal_Operculum):96 
    u:left_Area_Op9_(Frontal_Operculum):296
    u:right_Ch_123_(Basal_Forebrain):97 
    u:left_Ch_123_(Basal_Forebrain):297
    u:right_Area_6d1_(PreCG):98 
    u:left_Area_6d1_(PreCG):298
    u:right_Area_6d2_(PreCG):99 
    u:left_Area_6d2_(PreCG):299
    u:right_Area_6d3_(SFS):100
    u:left_Area_6d3_(SFS):300
    u:right_CM_(Amygdala):101
    u:left_CM_(Amygdala):301
    u:right_Area_hOc6_(POS):102
    u:left_Area_hOc6_(POS):302
    u:right_Area_hIP6_(IPS):103
    u:left_Area_hIP6_(IPS):303
    u:right_Area_hIP8_(IPS):104
    u:left_Area_hIP8_(IPS):304
    u:right_Area_hIP4_(IPS):105
    u:left_Area_hIP4_(IPS):305
    u:right_Area_hIP5_(IPS):106
    u:left_Area_hIP5_(IPS):306
    u:right_Area_hIP7_(IPS):107
    u:left_Area_hIP7_(IPS):307
    u:right_Area_hPO1_(POS):108
    u:left_Area_hPO1_(POS):308
    u:right_Area_6mp_(SMA,_mesial_SFG):109
    u:left_Area_6mp_(SMA,_mesial_SFG):309
    u:right_Area_6ma_(preSMA,_mesial_SFG):110
    u:left_Area_6ma_(preSMA,_mesial_SFG):310
    u:right_HC-Transsubiculum_(Hippocampus):111
    u:left_HC-Transsubiculum_(Hippocampus):311
    u:right_Tuberculum_(Basal_Forebrain):112
    u:left_Tuberculum_(Basal_Forebrain):312
    u:right_Terminal_islands_(Basal_Forebrain):113
    u:left_Terminal_islands_(Basal_Forebrain):313
    u:right_Area_Fo4_(OFC):114
    u:left_Area_Fo4_(OFC):314
    u:right_Area_Fo5_(OFC):115
    u:left_Area_Fo5_(OFC):315
    u:right_Area_Fo6_(OFC):116
    u:left_Area_Fo6_(OFC):316
    u:right_Area_Fo7_(OFC):117
    u:left_Area_Fo7_(OFC):317
    u:right_Area_8d1_(SFG):118
    u:left_Area_8d1_(SFG):318
    u:right_Area_8d2_(SFG):119
    u:left_Area_8d2_(SFG):319
    u:right_Area_8v2_(MFG):120
    u:left_Area_8v2_(MFG):320
    u:right_Area_8v1_(MFG):121
    u:left_Area_8v1_(MFG):321
    u:right_Area_Ig3_(Insula):122
    u:left_Area_Ig3_(Insula):322
    u:right_Area_Id4_(Insula):123
    u:left_Area_Id4_(Insula):323
    u:right_Area_Id5_(Insula):124
    u:left_Area_Id5_(Insula):324
    u:right_Area_Ia1_(Insula):125
    u:left_Area_Ia1_(Insula):325
    u:right_Area_Id6_(Insula):126
    u:left_Area_Id6_(Insula):326
    u:right_Area_SFS1_(SFS):127
    u:left_Area_SFS1_(SFS):327
    u:right_Area_SFS2_(SFS):128
    u:left_Area_SFS2_(SFS):328
    u:right_Area_MFG1_(MFG):129
    u:left_Area_MFG1_(MFG):329
    u:right_Area_MFG2_(MFG):130
    u:left_Area_MFG2_(MFG):330
    u:right_Area_Op5_(Frontal_Operculum):131
    u:left_Area_Op5_(Frontal_Operculum):331
    u:right_Area_Op6_(Frontal_Operculum):132
    u:left_Area_Op6_(Frontal_Operculum):332
    u:right_Area_Op7_(Frontal_Operculum):133
    u:left_Area_Op7_(Frontal_Operculum):333
    u:right_Area_Ph1_(PhG):134
    u:left_Area_Ph1_(PhG):334
    u:right_Area_Ph2_(PhG):135
    u:left_Area_Ph2_(PhG):335
    u:right_Area_Ph3_(PhG):136
    u:left_Area_Ph3_(PhG):336
    u:right_Area_CoS1_(CoS):137
    u:left_Area_CoS1_(CoS):337
    u:right_Medial_Accumbens,_ventral_Striatum_(Basal_Ganglia):138
    u:left_Medial_Accumbens,_ventral_Striatum_(Basal_Ganglia):338
    u:right_Fundus_of_Caudate_Nucleus,_ventral_Striatum_(Basal_Ganglia):139
    u:left_Fundus_of_Caudate_Nucleus,_ventral_Striatum_(Basal_Ganglia):339
    u:right_Fundus_of_Putamen,_Ventral_Striatum_(Basal_Ganglia):140
    u:left_Fundus_of_Putamen,_Ventral_Striatum_(Basal_Ganglia):340
    u:right_Lateral_Accumbens,_ventral_Striatum_(Basal_Ganglia):141
    u:left_Lateral_Accumbens,_ventral_Striatum_(Basal_Ganglia):341
    u:right_VP,_ventral_Pallidum_(Basal_Ganglia):142
    u:left_VP,_ventral_Pallidum_(Basal_Ganglia):342
    u:right_CGL_(Metathalamus):143
    u:left_CGL_(Metathalamus):343
    u:right_CGM_(Metathalamus):144
    u:left_CGM_(Metathalamus):344
    u:right_STN_(Subthalamus):145
    u:left_STN_(Subthalamus):345
    u:right_Area_Id9_(Insula):146
    u:left_Area_Id9_(Insula):346
    u:right_Area_Ia3_(Insula):147
    u:left_Area_Ia3_(Insula):347
    u:right_Area_Ia2_(Insula):148
    u:left_Area_Ia2_(Insula):348
    u:right_Area_Id8_(Insula):149
    u:left_Area_Id8_(Insula):349
    u:right_Area_Id10_(Insula):150
    u:left_Area_Id10_(Insula):350
    u:right_BST_(Bed_Nucleus):151
    u:left_BST_(Bed_Nucleus):351
    u:right_Frontal-I_(GapMap):152
    u:left_Frontal-I_(GapMap):352
    u:right_Frontal-II_(GapMap):153
    u:left_Frontal-II_(GapMap):353
    u:right_Temporal-to-Parietal_(GapMap):154
    u:left_Temporal-to-Parietal_(GapMap):354
    u:right_Frontal-to-Occipital_(GapMap):155
    u:left_Frontal-to-Occipital_(GapMap):355
    u:right_Frontal-to-Temporal-I_(GapMap):156
    u:left_Frontal-to-Temporal-I_(GapMap):356
    u:right_Frontal-to-Temporal-II_(GapMap):157
    u:left_Frontal-to-Temporal-II_(GapMap):357
    ----------- End regions for Julich_MNI2009c atlas --------------
    """)
    else:
        st.image("FS.afni.MNI2009c_asym_Atlas.png", caption="FS.afni.MNI2009c_asym Atlas")
        with st.expander("View FS.afni.MNI2009c_asym Region List"):
            st.text("""\
    Atlas FS.afni.MNI2009c_asym,      87 regions
    ----------- Begin regions for FS.afni.MNI2009c_asym atlas-----------
    u:Left-Cerebellum-Cortex:6  
    u:Left-Thalamus-Proper:7  
    u:Left-Caudate:8  
    u:Left-Putamen:9  
    u:Left-Pallidum:10 
    u:Brain-Stem:13 
    u:Left-Hippocampus:14 
    u:Left-Amygdala:15 
    u:Left-Accumbens-area:17 
    u:Left-VentralDC:18 
    u:Right-Cerebellum-Cortex:26 
    u:Right-Thalamus-Proper:27 
    u:Right-Caudate:28 
    u:Right-Putamen:29 
    u:Right-Pallidum:30 
    u:Right-Hippocampus:31 
    u:Right-Amygdala:32 
    u:Right-Accumbens-area:33 
    u:Right-VentralDC:34 
    u:ctx-lh-bankssts:47 
    u:ctx-lh-caudalanteriorcingulate:48 
    u:ctx-lh-caudalmiddlefrontal:49 
    u:ctx-lh-cuneus:50 
    u:ctx-lh-entorhinal:51 
    u:ctx-lh-fusiform:52 
    u:ctx-lh-inferiorparietal:53 
    u:ctx-lh-inferiortemporal:54 
    u:ctx-lh-isthmuscingulate:55 
    u:ctx-lh-lateraloccipital:56 
    u:ctx-lh-lateralorbitofrontal:57 
    u:ctx-lh-lingual:58 
    u:ctx-lh-medialorbitofrontal:59 
    u:ctx-lh-middletemporal:60 
    u:ctx-lh-parahippocampal:61 
    u:ctx-lh-paracentral:62 
    u:ctx-lh-parsopercularis:63 
    u:ctx-lh-parsorbitalis:64 
    u:ctx-lh-parstriangularis:65 
    u:ctx-lh-pericalcarine:66 
    u:ctx-lh-postcentral:67 
    u:ctx-lh-posteriorcingulate:68 
    u:ctx-lh-precentral:69 
    u:ctx-lh-precuneus:70 
    u:ctx-lh-rostralanteriorcingulate:71 
    u:ctx-lh-rostralmiddlefrontal:72 
    u:ctx-lh-superiorfrontal:73 
    u:ctx-lh-superiorparietal:74 
    u:ctx-lh-superiortemporal:75 
    u:ctx-lh-supramarginal:76 
    u:ctx-lh-frontalpole:77 
    u:ctx-lh-temporalpole:78 
    u:ctx-lh-transversetemporal:79 
    u:ctx-lh-insula:80 
    u:ctx-rh-bankssts:82 
    u:ctx-rh-caudalanteriorcingulate:83 
    u:ctx-rh-caudalmiddlefrontal:84 
    u:ctx-rh-cuneus:85 
    u:ctx-rh-entorhinal:86 
    u:ctx-rh-fusiform:87 
    u:ctx-rh-inferiorparietal:88 
    u:ctx-rh-inferiortemporal:89 
    u:ctx-rh-isthmuscingulate:90 
    u:ctx-rh-lateraloccipital:91 
    u:ctx-rh-lateralorbitofrontal:92 
    u:ctx-rh-lingual:93 
    u:ctx-rh-medialorbitofrontal:94 
    u:ctx-rh-middletemporal:95 
    u:ctx-rh-parahippocampal:96 
    u:ctx-rh-paracentral:97 
    u:ctx-rh-parsopercularis:98 
    u:ctx-rh-parsorbitalis:99 
    u:ctx-rh-parstriangularis:100
    u:ctx-rh-pericalcarine:101
    u:ctx-rh-postcentral:102
    u:ctx-rh-posteriorcingulate:103
    u:ctx-rh-precentral:104
    u:ctx-rh-precuneus:105
    u:ctx-rh-rostralanteriorcingulate:106
    u:ctx-rh-rostralmiddlefrontal:107
    u:ctx-rh-superiorfrontal:108
    u:ctx-rh-superiorparietal:109
    u:ctx-rh-superiortemporal:110
    u:ctx-rh-supramarginal:111
    u:ctx-rh-frontalpole:112
    u:ctx-rh-temporalpole:113
    u:ctx-rh-transversetemporal:114
    u:ctx-rh-insula:115
    ----------- End regions for FS.afni.MNI2009c_asym atlas --------------
    """)
    run_button = st.sidebar.button("Run Analysis")
    if run_button:
        if cluster_file_path is not None:
            with st.spinner("Running analysis..."):
                anatomical_str, interpretation, references = run_analysis(
                    cluster_file_path=cluster_file_path,
                    atlas=atlas,
                    task_description=task_description,
                    contrast_description=contrast_description,
                    use_ai=use_ai
                )
            # Store the results in session_state so they can be accessed later
            st.session_state.anatomical_str = anatomical_str
            st.session_state.interpretation = interpretation
            st.session_state.references = references 

            st.subheader(f"Deep Research Interpretation for the {max_clusters} clusters:")
            st.markdown(interpretation)
            if references:
                st.subheader("References")
                for i, ref in enumerate(references, start=1):
                    # Format: "- [1](URL): URL"
                    st.markdown(f"- {i}: {ref}")
            os.remove(cluster_file_path)
        else:
            st.warning("Please upload a cluster file (.1D) before running the analysis.")
    settings_summary = (
        f"Task Description: {task_description}\n"
        f"Contrast Description: {contrast_description}\n"
        f"Atlas: {atlas}\n"
        f"Coordinate System: {coord_system}"
    )

    report_filename = f"{sanitize_filename(task_description)}_{sanitize_filename(contrast_description)}_{sanitize_filename(atlas)}_Report.docx"

    # Only show the download button if the interpretation is available in session_state.
    if "interpretation" in st.session_state and st.session_state.interpretation:
        # Generate the Word report.
        my_references = st.session_state.get("references", [])

        report_path = create_word_report(
            settings_summary,
            st.session_state.anatomical_str,
            st.session_state.interpretation,
            my_references
        )

        with open(report_path, "rb") as f:
            report_bytes = f.read()
        st.download_button(
            label="Download Report as Word Document",
            data=report_bytes,
            file_name=report_filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        # Optionally, remove the temporary file after download.
        os.remove(report_path)
else:
    # When SPM is selected, only the conversion is performed
    if "converted_file_content" in st.session_state:
        st.markdown("### Main Clusters Detected and Converted into a .1D File:")
        st.code(st.session_state.converted_file_content, language="text")

        # Provide a download button for the converted .1D file directly in the browser
        st.download_button(
            label="Download Converted .1D File",
            data=file_content,
            file_name=output_filename,
            mime="text/plain"
        )
    else:
        st.warning("Please upload a SPM .m file to convert to .1D file in order to be analyzed.")


st.markdown(
    """
    <style>
    .credits-discrete {
        font-size: 0.85rem; /* Slightly smaller than normal text */
        color: #888;        /* Gray text */
        text-align: right;  /* Align to the right side of its container */
        margin-right: 1rem; /* Optional spacing from the edge */
    }
    .highlight-name {
        text-decoration: underline; /* Underline the name */
    }
    </style>
    <div class="credits-discrete">
      Credits to 
      <a href="https://github.com/tdelavy/fMRI-ClusterAI" 
         target="_blank"
         style="color: inherit; text-decoration: none;">
        <span class="highlight-name">Thibaud Delavy</span>
      </a>
    </div>
    """,
    unsafe_allow_html=True
)
