import warnings
import tables, click
import numpy as np
from tqdm import tqdm
from heatmaps_common import load_exac_data, load_sim_data, load_filtered_df

LIKELIHOODS = ['prf', 'kde', 'kde_nearest']
DEMOGRAPHIES = ['tennessen', 'supertennessen']
S_VALUES = ['NEUTRAL', '-4.0', '-3.0', '-2.0', '-1.0']
H_VALUES = [0.0, 0.1, 0.3, 0.5]
FUNCS = ['LOF_probably', 'synon']
GENESETS = ['all', 'haplo_Hurles_80', 'CGD_AD', 'inbred_ALL', 'haplo_Hurles_low20', 'CGD_AR']

@click.command()
@click.option('--output', '-o', type=click.Path(writable=True, dir_okay=False), default='-')
def main(output):
    with tables.open_file(output, mode='w', title='heatmaps') as h5file:
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
        for spec in tqdm(heatmaps_to_load):
            key = "/" + "/".join(str(arg) for arg in spec)
            kind, *args = spec
            if kind == "simulated_single":
                histogram = load_sim_data(*args)
            elif kind == "simulated_geneset":
                likelihood, ref, sim, s, h, func, geneset, min_L, max_L = args
                filtered_df = load_filtered_df(ref, func, geneset, likelihood, min_L, max_L)
                L = filtered_df.U.transform('log10') + 8.0
                histogram = load_sim_data(likelihood, ref, sim, s, h, L)
            elif kind == "exac":
                histogram = load_exac_data(*args)
            else:
                raise ValueError(f"Unrecognized heatmap kind {kind!r}")
            histogram = np.array(histogram, dtype=float)
            frac = histogram / np.nansum(histogram)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", tables.NaturalNameWarning)
                h5file.create_array(key, 'histogram', histogram, createparents=True)
                h5file.create_array(key, 'frac', frac)

if __name__ == "__main__":
    main()
