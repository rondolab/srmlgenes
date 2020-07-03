import warnings
import tables, click
import numpy as np
from tqdm import tqdm
import heatmaps_common
from heatmaps_common import load_exac_data, load_sim_data, load_filtered_df, LIKELIHOODS, DEMOGRAPHIES, S_VALUES, \
    H_VALUES, FUNCS, GENESETS, LIKELIHOOD_ENUM, DEMOGRAPHY_ENUM, S_ENUM, H_ENUM, FUNC_ENUM, GENESET_ENUM, \
    SimulationHeatmapFixedLength, SimulationHeatmapVariableLength, EmpiricalHeatmap
from joblib import Parallel, delayed


@click.command()
@click.option('--output', '-o', type=click.Path(writable=True, dir_okay=False), default=heatmaps_common.HEATMAP_TABLES_PATH)
@click.option('--n-jobs', '-j', type=int, envvar="LSB_DJOB_NUMPROC", default=1)
@click.option("--truncate", type=int, default=None)
@click.option("--include-single-sims/--skip-single-sims", default=True)
@click.option("--include-geneset-sims/--skip-geneset-sims", default=True)
@click.option("--include-exac/--skip-exac", default=True)
def main(output, n_jobs, truncate, include_single_sims, include_geneset_sims, include_exac):
    heatmaps_to_load = []
    with tqdm(desc="collecting parameters") as progress:
        for likelihood in LIKELIHOODS:
            for ref_demography in DEMOGRAPHIES:
                for sim_demography in DEMOGRAPHIES:
                    for s in S_VALUES:
                        for h in H_VALUES:
                            if s == "NEUTRAL" and h != "0.5":
                                continue
                            if include_single_sims:
                                for L in np.arange(2.0, 5.1, 0.1).round(1):
                                    if include_single_sims:
                                        heatmaps_to_load.append(('simulated_single', likelihood, ref_demography, sim_demography, s, h, L))
                                        progress.update(1)
                            if include_geneset_sims:
                                for func in FUNCS:
                                    for geneset in GENESETS:
                                        for min_L in np.arange(2.0, 5.0, 0.1).round(1):
                                            for max_L in np.arange(min_L + 0.1, 5.1, 0.1).round(1):
                                                heatmaps_to_load.append(('simulated_geneset', likelihood, ref_demography, sim_demography, s, h, func, geneset, min_L, max_L))
                                                progress.update(1)
                if include_exac:
                    for func in FUNCS:
                        for geneset in GENESETS:
                            for min_L in np.arange(0.0, 6.0, 0.1).round(1):
                                for max_L in np.arange(min_L + 0.1, 6.1, 0.1).round(1):
                                    heatmaps_to_load.append(('exac', likelihood, ref_demography, func, geneset, min_L, max_L))
                                    progress.update(1)
    if truncate is not None:
        heatmaps_to_load = heatmaps_to_load[:truncate]
    if output == heatmaps_common.HEATMAP_TABLES_PATH:
        heatmaps_common.HEATMAP_TABLES_FILE.close()
    with tables.open_file(output, mode='w', title='heatmaps', filters=tables.Filters(complevel=6, complib="blosc")) as h5file:
        heatmaps_group = h5file.create_group(h5file.root, "heatmaps")
        h5file.create_table(heatmaps_group, "simulated_single", SimulationHeatmapFixedLength, expectedrows=15000)
        h5file.create_table(heatmaps_group, "simulated_geneset", SimulationHeatmapVariableLength, expectedrows=3000000)
        h5file.create_table(heatmaps_group, "exac", EmpiricalHeatmap, expectedrows=50000)
        for table_name, row in Parallel(n_jobs=n_jobs, backend="multiprocessing")(delayed(load_heatmap)(*args) for args in tqdm(heatmaps_to_load, desc="calculating heatmaps")):
            getattr(heatmaps_group, table_name).append([row])
        # index tables
        for col_name in tqdm(["likelihood", "ref_demography", "sim_demography", "s", "h", "L"],
                             desc="indexing simulated_single table"):
            heatmaps_group.simulated_single.colinstances[col_name].create_index()
        for col_name in tqdm(["likelihood", "ref_demography", "sim_demography", "s", "h",
                              "func", "geneset", "min_L", "max_L"],
                             desc="indexing simulated_geneset table"):
            heatmaps_group.simulated_geneset.colinstances[col_name].create_index()
        for col_name in tqdm(["likelihood", "ref_demography",
                              "func", "geneset", "min_L", "max_L"],
                             desc="indexing exac table"):
            heatmaps_group.exac.colinstances[col_name].create_index()


def load_heatmap(kind, *args):
    if kind == "simulated_single":
        likelihood, ref, sim, s, h, L = args
        record = (LIKELIHOOD_ENUM[likelihood],
                  DEMOGRAPHY_ENUM[ref],
                  DEMOGRAPHY_ENUM[sim],
                  S_ENUM[s],
                  H_ENUM[h],
                  float(L))
        histogram = load_sim_data(*args)
        excess_results = ()
    elif kind == "simulated_geneset":
        likelihood, ref, sim, s, h, func, geneset, min_L, max_L = args
        record = (LIKELIHOOD_ENUM[likelihood],
                  DEMOGRAPHY_ENUM[ref],
                  DEMOGRAPHY_ENUM[sim],
                  S_ENUM[s],
                  H_ENUM[h],
                  FUNC_ENUM[func],
                  GENESET_ENUM[geneset],
                  float(min_L), float(max_L))

        filtered_df = load_filtered_df(ref, func, geneset, likelihood, min_L, max_L)
        L = filtered_df.U.transform('log10') + 8.0
        try:
            histogram = load_sim_data(likelihood, ref, sim, s, h, L)
        except ValueError:
            histogram = get_null_histogram()
        excess_results = ()
    elif kind == "exac":
        likelihood, demography, func, geneset, min_L, max_L = args
        record = (LIKELIHOOD_ENUM[likelihood],
                  DEMOGRAPHY_ENUM[demography],
                  FUNC_ENUM[func],
                  GENESET_ENUM[geneset],
                  float(min_L), float(max_L))
        try:
            histogram, odds_ratios, p_values = load_exac_data(likelihood, demography, func, geneset, min_L, max_L)
        except ValueError:
            histogram = get_null_histogram()
            odds_ratios = get_null_histogram()
            p_values = get_null_histogram()
        excess_results = (odds_ratios, p_values)
    else:
        raise ValueError(f"Unrecognized heatmap kind {kind!r}")
    histogram = np.array(histogram, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = histogram / np.nansum(histogram)
    return kind, record + (histogram, frac) + excess_results


def get_null_histogram():
    histogram = np.zeros((4,5))
    histogram[:-1,0] = np.nan
    return histogram

if __name__ == "__main__":
    main()
