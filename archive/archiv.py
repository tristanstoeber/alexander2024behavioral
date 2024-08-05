path_exp = "/home/docker/data/PNN.A-CMWMRawData/"
path_probe_trials = "PNN.A-C.MWM.probe/"
fname_genotypes = "MWMGenotypes.xlsx"

df_info = load_finfo_probe(path_exp+path_probe_trials)

df_genotypes = pd.read_excel(path_exp+fname_genotypes)
