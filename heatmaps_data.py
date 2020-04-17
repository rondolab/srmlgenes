import os
from functools import lru_cache

import numpy as np
import pandas as pd
from plotly import graph_objects as go

SIM_DATA_TEMPLATE = "~/genecad/04_dominance/genedoses/simulation_inference_{likelihood}/{likelihood}_ref_{ref}_sims_{sim}_S_{s}_h_{h}_L_{L}.tsv"
EXAC_SUMSTATS_TABLE = pd.read_table("/hpc/users/jordad05/genecad/04_dominance/genedose/ExAC_63K_symbol_plus_ensembl_func_summary_stats.tsv")

FUNC_LENGTH_TABLES = {}
for func in 'LOF_probably', 'synon':
    FUNC_LENGTH_TABLES[func] = EXAC_SUMSTATS_TABLE.loc[EXAC_SUMSTATS_TABLE.func == func, "L"]\
                                            .transform('log10')\
                                            .round(1)\
                                            .clip(2.0, 5.0)\
                                            .value_counts()

LIKELIHOOD_FILES = {('kde_nearest', 'tennessen') : "ExAC_kde_inference_nearest.20200130.tsv",
                    ('kde_nearest', 'supertennessen') : "ExAC_63K_kde_nearest_nolscale_supertennessen_inference.tsv",
                    ('kde_nearest', 'subtennessen') : "kde_nearest_nolscale_exac_inference_subtennessen.tsv",
                    ('kde_3d', 'tennessen') : "ExAC_kde_inference_3d.20200124.tsv",
                    ('kde_3d', 'supertennessen') : "ExAC_63K_kde_3d_nolscale_supertennessen_inference.tsv",
                    ('kde_3d', 'subtennessen') : "kde_3d_nolscale_exac_inference_subtennessen.tsv",
                    ('prf', 'tennessen') : "ExAC_prf_inference.20191212.tsv",
                    ('prf', 'supertennessen') : "ExAC_63K_prf_supertennessen_inference.tsv",
                    ('prf', 'subtennessen') : "ExAC_63K_prf_subtennessen_inference.tsv"}
BASE_DIR = "/sc/arion/projects/GENECAD/04_dominance"

genesets = {}
for name in "CGD_AR", "CGD_AD", "inbred_ALL", "haplo_Hurles_80", "haplo_Hurles_low20":
    filename = os.path.join(BASE_DIR, 'slim/mock_genome', name + '.tsv')
    geneset = set()
    with open(filename) as list_file:
        for gene in list_file:
            if gene.strip() != "gene":
                geneset.add(gene.strip())
    genesets[name] = geneset



def format_heatmap_sims(df):
    ml_bin_names = df.transpose().drop("L").idxmax()
    split_names = ml_bin_names.str.split("_")
    ml_s = split_names.str.get(0)
    ml_h = split_names.str.get(1)
    crosstab = pd.crosstab(ml_h, ml_s)
    counts_grid = crosstab.reindex(["0.0", "0.1", "0.3", "0.5"], fill_value=0, axis=0)\
                          .reindex(["NEUTRAL", "-4.0", "-3.0", "-2.0", "-1.0"], fill_value=0, axis=1)\
                          .values.tolist()
    for h_index in 0, 1, 2:
        counts_grid[h_index][0] = None
    return counts_grid


@lru_cache(maxsize=None)
def load_sim_data(likelihood, ref, sim, s, h, L):
    if L in FUNC_LENGTH_TABLES:
        sims_to_concat = []
        for l, count in FUNC_LENGTH_TABLES[L].iteritems():
            sims_to_concat.append(pd.read_table(SIM_DATA_TEMPLATE.format(likelihood=likelihood,
                                                                         ref=ref,
                                                                         sim=sim,
                                                                         s=s,
                                                                         h=h,
                                                                         L=l),
                                                nrows=count))
        df = pd.concat(sims_to_concat, ignore_index=True)
    else:
        df = pd.read_table(SIM_DATA_TEMPLATE.format(likelihood=likelihood,
                                                    ref=ref,
                                                    sim=sim,
                                                    s=s, h=h, L=L))
    return format_heatmap_sims(df)


def heatmap_figure(heatmap_data):
    heatmap_data = np.array(heatmap_data, dtype=float)
    total_genes = np.nansum(heatmap_data)
    percent = np.round(heatmap_data / total_genes * 100, 1)
    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_data,
                        x=['Neutral', '-10⁻⁴', '-10⁻³', '-10⁻²', '-10⁻¹'],
                        y=["0.0", "0.1", "0.3", "0.5"],
                        customdata=percent,
                        hoverongaps=False,
                        hovertemplate=f"h: %{{y}}<br />s: %{{x}}<br />genes: %{{z}}/{total_genes:0.0f} (%{{customdata}}%)<extra></extra>"),
                    layout=go.Layout(width=800, height=600,
                xaxis_type='category', yaxis_type='category'))
    return fig


@lru_cache(maxsize=None)
def load_unfiltered_df(likelihood, demography):
    filename = os.path.join(BASE_DIR, "genedose", LIKELIHOOD_FILES[likelihood, demography])
    return pd.read_table(filename)


def filter_df(df, func, genelist, min_L, max_L):
    selector = df.func == func
    selector &= df.U.between(10**(min_L-8), 10**(max_L-8))
    if genelist != "all":
        selector &= df.gene.isin(genesets[genelist])
    return df.loc[selector]


def format_heatmap_empirical(filtered_df):
    crosstab = pd.crosstab(filtered_df["h_grid17_ml"],
                           filtered_df["s_grid17_ml"],
                           dropna=False, margins=True)
    counts_grid = []
    for h in 0.0, 0.1, 0.3, 0.5:
        if h == 0.5:
            counts_row = [crosstab.loc["All", 0.0]]
        else:
            counts_row = [None]
        for s in 1e-4, 1e-3, 1e-2, 1e-1:
            try:
                counts_row.append(crosstab.loc[h, s])
            except KeyError:
                counts_row.append(0)
        counts_grid.append(counts_row)
    return counts_grid


@lru_cache(maxsize=None)
def load_exac_data(likelihood, demography, func, genelist, min_L, max_L):
    unfiltered_df = load_unfiltered_df(likelihood, demography)
    filtered_df = filter_df(unfiltered_df, func, genelist, min_L, max_L)
    return format_heatmap_empirical(filtered_df)


