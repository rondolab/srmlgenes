import os
import warnings
from functools import lru_cache, wraps

import numpy as np
import pandas as pd
from plotly import graph_objects as go
import mpmath

BASE_DIR = os.path.join(os.path.dirname(__file__), "dominance_data")
SIM_DATA_TEMPLATE = os.path.join(BASE_DIR, "sims",
                                 "{likelihood}_ref_{ref}_sims_{sim}_S_{s}_h_{h}_L_{L:.1f}.tsv")

LIKELIHOOD_FILE = "ExAC_63K_prf_supertennessen_inference.tsv"
HIQUAL_GENESET_NAME = "clinvar_atleast2_2plus"
GENESETS = ['all', 'haplo_Hurles_80', 'CGD_AD_2020', 'inbred_ALL', 'haplo_Hurles_low20', 'CGD_AR_2020',
            HIQUAL_GENESET_NAME]

GENESETS_DICT = {}


class DataFileWarning(UserWarning):
    pass


geneset_base_dir = os.path.join(BASE_DIR, "genesets")

try:
    for geneset_name in filter(lambda name: name != "all", GENESETS):
        filename = os.path.join(geneset_base_dir, geneset_name + '.tsv')
        geneset = set()
        with open(filename) as list_file:
            for gene in list_file:
                if gene.strip() != "gene":
                    geneset.add(gene.strip())
        GENESETS_DICT[geneset_name] = geneset
except FileNotFoundError:
    warnings.warn(f"Gene list files not found (looking in {geneset_base_dir})", DataFileWarning)


def extract_histogram_sims(df):
    ml_bin_names = df.transpose().drop("L").idxmax()
    split_names = ml_bin_names.str.split("_")
    ml_s = split_names.str.get(0)
    ml_h = split_names.str.get(1)
    crosstab = pd.crosstab(ml_h, ml_s)
    counts_grid = crosstab.reindex(["0.0", "0.1", "0.3", "0.5"], fill_value=0, axis=0)\
                          .reindex(["NEUTRAL", "-4.0", "-3.0", "-2.0", "-1.0"], fill_value=0, axis=1)\
                          .astype(float).values
    for h_index in 0, 1, 2:
        counts_grid[h_index, 0] = np.nan
    return counts_grid


def tuplify_args(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if not np.isscalar(arg):
                new_arg = tuple(arg)
            else:
                new_arg = arg
            new_args.append(new_arg)
        new_kwargs = {}
        for key, value in kwargs.items():
            if not np.isscalar(arg):
                new_value = tuple(value)
            else:
                new_value = value
            new_kwargs[key] = new_value
        return f(*new_args, **new_kwargs)
    return wrapper


@tuplify_args
@lru_cache(maxsize=None)
def load_sim_data(likelihood, ref, sim, s, h, L):
    if isinstance(L, tuple):
        L = pd.Series(L)
        sims_to_concat = []
        for l, count in L.round(1).value_counts().iteritems():
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
    return extract_histogram_sims(df)


def heatmap_figure(heatmap_data_row, z_variable="histogram"):
    total_genes = np.nansum(heatmap_data_row["histogram"])
    if z_variable == "histogram":
        z = heatmap_data_row["histogram"]
        zmin = 0
        zmax = total_genes*0.3
        try:
            customdata = np.dstack((heatmap_data_row["frac"],
                                    heatmap_data_row["odds_ratios"],
                                    heatmap_data_row["p_values"]))
            hovertemplate = f"""h: %{{y}}<br />
s: %{{x}}<br />
genes: %{{z}} / {total_genes:0.0f} (%{{customdata[0]:.1%}})<br />
enrichment: %{{customdata[1]:0.2f}} (p-value = %{{customdata[2]:0.2g}})<extra></extra>"""
        except (ValueError, KeyError):
            customdata = heatmap_data_row["frac"]
            hovertemplate = f"""h: %{{y}}<br />
s: %{{x}}<br />
genes: %{{z}} / {total_genes:0.0f} (%{{customdata:.1%}})<extra></extra>"""
        extra_args = {}
    else:
        customdata = np.dstack((heatmap_data_row["histogram"],
                                heatmap_data_row["frac"],
                                heatmap_data_row["odds_ratios"],
                                heatmap_data_row["p_values"]))
        hovertemplate = f"""h: %{{y}}<br />
s: %{{x}}<br />
genes: %{{customdata[0]}} / {total_genes:0.0f} (%{{customdata[1]:.1%}})<br />
enrichment: %{{customdata[2]:0.2f}} (p-value = %{{customdata[3]:0.2g}}) <extra></extra>"""
        extra_args = { 'colorscale': 'RdBu', 'zmid': 0}
        with np.errstate(divide="ignore"):
            log_oddsratio = np.log(heatmap_data_row["odds_ratios"])
            log_oddsratio[log_oddsratio == -np.inf] = -10.0

        if z_variable == "p_value":
            z = np.copysign(np.log10(heatmap_data_row["p_values"]), log_oddsratio)
            zmin = -10.0
            zmax = 10.0
        elif z_variable == "odds_ratio":
            z = log_oddsratio
            zmin = -1.5
            zmax = 1.5
        else:
            raise ValueError(f"Unrecognized z variable {z_variable}")

    fig = go.Figure(data=go.Heatmap(
                        z=z,
                        x=['Neutral', '-10⁻⁴', '-10⁻³', '-10⁻²', '-10⁻¹'],
                        y=["0.0", "0.1", "0.3", "0.5"],
                        zmin=zmin,
                        zmax=zmax,
                        customdata=customdata,
                        hoverongaps=False,
                        hovertemplate=hovertemplate,
                        **extra_args),
                    layout=go.Layout(width=800, height=600,
                    xaxis_type='category', yaxis_type='category'))
    return fig


@lru_cache(maxsize=None)
def load_unfiltered_df(likelihood, demography):
    filename = os.path.join(BASE_DIR, LIKELIHOOD_FILE)
    return pd.read_table(filename)


def filter_df(df, func, genelist, quality, min_L, max_L):
    selector = df.func == func
    selector &= df.U.between(10**(min_L-8), 10**(max_L-8))
    if genelist != "all":
        if isinstance(genelist, (set, frozenset)):
            selector &= df.gene.isin(genelist)
        else:
            selector &= df.gene.isin(GENESETS_DICT[genelist])
    if quality == "high":
        selector &= df.gene.isin(GENESETS_DICT[HIQUAL_GENESET_NAME])
    elif quality == "low":
        selector &= ~df.gene.isin(GENESETS_DICT[HIQUAL_GENESET_NAME])
    return df.loc[selector]


def extract_histogram_empirical(filtered_df):
    crosstab = pd.crosstab(filtered_df["h_grid17_ml"],
                           filtered_df["s_grid17_ml"],
                           dropna=False, margins=True)
    counts_grid = []
    for h in 0.0, 0.1, 0.3, 0.5:
        if h == 0.5:
            try:
                counts_row = [crosstab.loc["All", 0.0]]
            except KeyError:
                counts_row = [0]
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
def load_exac_data(likelihood, demography, func, geneset, quality, min_L, max_L):
    geneset_df = load_filtered_df(demography, func, geneset, quality, likelihood, min_L, max_L)
    geneset_histogram = np.array(extract_histogram_empirical(geneset_df), dtype=float)
    geneset_count = len(geneset_df)
    if geneset == "all":
        ones = np.ones_like(geneset_histogram)
        ones[np.isnan(geneset_histogram)] = np.nan
        return geneset_histogram, ones, ones
    all_df = load_filtered_df(demography, func, "all", "all", likelihood, min_L, max_L)
    all_count = len(all_df)
    all_histogram = np.array(extract_histogram_empirical(all_df), dtype=float)
    complement_histogram = all_histogram - geneset_histogram
    complement_count = all_count - geneset_count

    # contingency tables
    a_ary = complement_count - complement_histogram # not in geneset and not in bin
    b_ary = geneset_count - geneset_histogram # in geneset but not in bin
    c_ary = complement_histogram # in bin but not in bin
    d_ary = geneset_histogram # in geneset and in bin

    with np.errstate(divide='ignore', invalid='ignore'):
        odds_ratios = (a_ary * d_ary) / (b_ary * c_ary)

    with np.nditer([a_ary, b_ary, c_ary, d_ary, None]) as it:
        for a, b, c, d, p_value in it:
            table = np.array([[a,b], [c,d]])
            expected =  table.sum(axis=1)[:,np.newaxis] * \
                        table.sum(axis=0)[np.newaxis,:] / \
                        table.sum()
            chi2 = np.sum((table - expected)**2 / expected)
            p = mpmath.gammainc(1/2, chi2/2, regularized=True)
            p_value[...] = p
        p_values = it.operands[4]

    return geneset_histogram, odds_ratios, p_values


def load_filtered_df(demography, func, genelist, quality, likelihood, min_L, max_L):
    unfiltered_df = load_unfiltered_df(likelihood, demography)
    filtered_df = filter_df(unfiltered_df, func, genelist, quality, min_L, max_L)
    return filtered_df


def get_null_histogram():
    histogram = np.zeros((4,5))
    histogram[:-1,0] = np.nan
    return histogram


def make_heatmap_single_sim(likelihood, ref, sim, s, h, L):
    histogram = load_sim_data(likelihood, ref, sim, s, h, L)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = histogram / np.nansum(histogram)
    return heatmap_figure({'histogram': histogram,
                           'frac': frac})


def make_heatmap_geneset_sim(likelihood, ref, sim, s, h, func, geneset, quality, min_L, max_L):
    filtered_df = load_filtered_df(ref, func, geneset, quality, likelihood, min_L, max_L)
    L = filtered_df.U.transform('log10') + 8.0
    try:
        histogram = load_sim_data(likelihood, ref, sim, s, h, L)
    except ValueError:
        histogram = get_null_histogram()
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = histogram / np.nansum(histogram)
    return heatmap_figure({'histogram': histogram,
                           "frac": frac})


def make_heatmap_empirical(likelihood, demography, func, genelist, quality, min_L, max_L, z_variable="histogram"):
    try:
        histogram, odds_ratio, p_value = load_exac_data(likelihood, demography, func, genelist, quality, min_L, max_L)
    except ValueError:
        histogram = get_null_histogram()
        odds_ratio = get_null_histogram()
        p_value = get_null_histogram()
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = histogram / np.nansum(histogram)
    return heatmap_figure({"histogram": histogram,
                           "frac": frac,
                           "odds_ratios": odds_ratio,
                           "p_values": p_value}, z_variable)
