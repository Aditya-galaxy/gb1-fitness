# PyMOL script: GB1 and Antibody Fc binding interface
load data/1FCC.pdb, complex
hide everything, complex
show cartoon, complex

# Color chains
color cyan, chain C or chain D     # GB1 domains
color wheat, chain A or chain B    # Human IgG Fc heavy chains

# Highlight the 4 combinatorial mutation sites on GB1
select mut_sites, chain C and resi 39+40+41+54
show sticks, mut_sites
color magenta, mut_sites and name C*
set stick_radius, 0.25, mut_sites

# Highlight key contacting Fc residues
select fc_epitope, chain A and resi 433+434
show sticks, fc_epitope
color yellow, fc_epitope and name C*

# Show hydrogen bonds / polar contacts
distance hbonds, mut_sites, fc_epitope, mode=2
set dash_color, red, hbonds
set dash_width, 2.0, hbonds

zoom mut_sites, 8.0
set ray_shadow, 0
bg_color white
