import warnings
import tables, click
import numpy as np
from tqdm import tqdm
from heatmaps_common import load_exac_data, load_sim_data, load_filtered_df
from joblib import Parallel, delayed

LIKELIHOODS = ['prf', 'kde', 'kde_nearest']
DEMOGRAPHIES = ['tennessen', 'supertennessen']
S_VALUES = ['NEUTRAL', '-4.0', '-3.0', '-2.0', '-1.0']
H_VALUES = [0.0, 0.1, 0.3, 0.5]
FUNCS = ['LOF_probably', 'synon']
GENESETS = ['all', 'haplo_Hurles_80', 'CGD_AD', 'inbred_ALL', 'haplo_Hurles_low20', 'CGD_AR']

@click.command()
@click.option('--output', '-o', type=click.Path(writable=True, dir_okay=False), default='-')
@click.option('--n-jobs', '-j', type=int, default=1)
def main(output, n_jobs):
    heatmaps_to_load = []
    for likelihood in LIKELIHOODS:
        for ref_demography in DEMOGRAPHIES:
            for sim_demography in DEMOGRAPHIES:
                for s in S_VALUES:
                    for h in H_VALUES:
                        if s == "NEUTRAL" and h != 0.5:
                            continue
                        for L in np.arange(2.0, 5.1, 0.1).round(1):
                            heatmaps_to_load.append(('simulated_single', likelihood, ref_demography, sim_demography, s, h, L))
                        for func in FUNCS:
                            for geneset in GENESETS:
                                for min_L in np.arange(2.0, 5.0, 0.1).round(1):
                                    for max_L in np.arange(min_L + 0.1, 5.1, 0.1).round(1):
                                        heatmaps_to_load.append(('simulated_geneset', likelihood, ref_demography, sim_demography, s, h, func, geneset, min_L, max_L))
            for func in FUNCS:
                for geneset in GENESETS:
                    for min_L in np.arange(0.0, 6.0, 0.1).round(1):
                        for max_L in np.arange(min_L + 0.1, 6.1, 0.1).round(1):
                            heatmaps_to_load.append(('exac', likelihood, ref_demography, func, geneset, min_L, max_L))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", tables.NaturalNameWarning)
        with tables.open_file(output, mode='w', title='heatmaps') as h5file:
            for key, results in Parallel(n_jobs=n_jobs)(delayed(load_heatmap)(*args) for args in tqdm(heatmaps_to_load)):
                group = h5file.create_group(h5file.root, key, createparents=True)
                for name, table in results.items():
                    h5file.create_array(group, name, table)

def null_heatmap():
    return [ [0.0,  0.0, 0.0, 0.0],
             [None, 0.0, 0.0, 0.0],
             [None, 0.0, 0.0, 0.0],
             [None, 0.0, 0.0, 0.0] ]


def load_heatmap(kind, *args):
    key = f"/{kind}" + "/".join(str(arg) for arg in args)
    if kind == "simulated_single":
        histogram = load_sim_data(*args)
    elif kind == "simulated_geneset":
        likelihood, ref, sim, s, h, func, geneset, min_L, max_L = args
        filtered_df = load_filtered_df(ref, func, geneset, likelihood, min_L, max_L)
        L = filtered_df.U.transform('log10') + 8.0
        try:
            histogram = load_sim_data(likelihood, ref, sim, s, h, L)
        except ValueError:
            histogram = null_heatmap()
    elif kind == "exac":
        try:
            histogram = load_exac_data(*args)
        except ValueError:
            histogram = null_heatmap()
    else:
        raise ValueError(f"Unrecognized heatmap kind {kind!r}")
    histogram = np.array(histogram, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = histogram / np.nansum(histogram)
    return key, {   'histogram' : histogram, 
                    'frac' : frac }


if __name__ == "__main__":
    main()
