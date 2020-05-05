import os
from functools import lru_cache

import numpy as np
import pandas as pd
import tables
from plotly import graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html

from sim import tables_file

from exac import tables_file

SIM_DATA_TEMPLATE = "~/genecad/04_dominance/genedose/simulation_inference_{likelihood}/{likelihood}_ref_{ref}_sims_{sim}_S_{s}_h_{h}_L_{L}.tsv"
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
                    ('kde', 'tennessen') : "ExAC_kde_inference_3d.20200124.tsv",
                    ('kde', 'supertennessen') : "ExAC_63K_kde_3d_nolscale_supertennessen_inference.tsv",
                    ('kde', 'subtennessen') : "kde_3d_nolscale_exac_inference_subtennessen.tsv",
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


#@lru_cache(maxsize=None)
def load_sim_data(likelihood, ref, sim, s, h, L):
    if isinstance(L, pd.Series):
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
    return format_heatmap_sims(df)


def heatmap_figure(heatmap_data_group):
    total_genes = np.nansum(heatmap_data_group.histogram)
    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_data_group.histogram,
                        x=['Neutral', '-10⁻⁴', '-10⁻³', '-10⁻²', '-10⁻¹'],
                        y=["0.0", "0.1", "0.3", "0.5"],
                        customdata=heatmap_data_group.frac,
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
    filtered_df = load_filtered_df(demography, func, genelist, likelihood, max_L, min_L)
    return format_heatmap_empirical(filtered_df)


def load_filtered_df(demography, func, genelist, likelihood, min_L, max_L):
    unfiltered_df = load_unfiltered_df(likelihood, demography)
    filtered_df = filter_df(unfiltered_df, func, genelist, min_L, max_L)
    return filtered_df


def create_app(app_name, app_filename):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(app_name, external_stylesheets=external_stylesheets,
                    requests_pathname_prefix="/" + os.path.relpath(app_filename, "/hpc/web/users.hpc.mssm.edu") + "/")
    if "dev" in __file__:
        app.enable_dev_tools(debug=True)
    return app

def gene_select_controls():
    return [
        dcc.Dropdown(id="geneset-dropdown",
                     options=[{'label': 'All genes', 'value': 'all'},
                              {'label': 'HI > 80%', 'value': 'haplo_Hurles_80'},
                              {'label': 'CGD AD', 'value': 'CGD_AD'},
                              {'label': 'Inbred', 'value': 'inbred_ALL'},
                              {'label': 'HI < 20%', 'value': 'haplo_Hurles_low20'},
                              {'label': 'CGD AR', 'value': 'CGD_AR'}],
                     value='all', style={'width': '7em', 'display': 'inline-block'}),
        dcc.Dropdown(id="func-dropdown",
                     options=[{'label': "LOF + PolyPhen probably", 'value': 'LOF_probably'},
                              {'label': "synonymous", 'value': 'synon'}],
                     value="LOF_probably", style={'width': '15em', 'display': 'inline-block'}),
        html.Label("L"), dcc.RangeSlider(id="L-slider", min=0, max=6, step=0.1,
                                     marks={0: '10⁰', 1: '10¹', 2: '10²', 3: '10³', 4: '10⁴', 5: '10⁵', 6: '10⁶'},
                                     value=[2, 5],
                                     tooltip={'always_visible': False})
    ]


LIKELIHOODS = ['prf', 'kde', 'kde_nearest']
DEMOGRAPHIES = ['tennessen', 'supertennessen']
S_VALUES = ['NEUTRAL', '-4.0', '-3.0', '-2.0', '-1.0']
H_VALUES = ['0.0', '0.1', '0.3', '0.5']
FUNCS = ['LOF_probably', 'synon']
GENESETS = ['all', 'haplo_Hurles_80', 'CGD_AD', 'inbred_ALL', 'haplo_Hurles_low20', 'CGD_AR']
LIKELIHOOD_ENUM = tables.Enum(LIKELIHOODS)
DEMOGRAPHY_ENUM = tables.Enum(DEMOGRAPHIES)
S_ENUM = tables.Enum(S_VALUES)
H_ENUM = tables.Enum(H_VALUES)
FUNC_ENUM = tables.Enum(FUNCS)
GENESET_ENUM = tables.Enum(GENESETS)
BASE_OFFSET = 0
SIM_OFFSET = 10
GENE_OFFSET = 20
DATA_OFFSET = 30


class HeatmapBase(tables.IsDescription):
    histogram = tables.Float64Col(shape=(4,5), pos=DATA_OFFSET + 0)
    frac = tables.Float64Col(shape=(4,5), pos=DATA_OFFSET + 1)
    likelihood = tables.EnumCol(LIKELIHOOD_ENUM, "prf", base='uint8', pos=BASE_OFFSET + 0)
    ref_demography = tables.EnumCol(DEMOGRAPHY_ENUM, "tennessen", base='uint8', pos=BASE_OFFSET + 1)


class SimulationHeatmapBase(HeatmapBase):
    sim_demography = tables.EnumCol(DEMOGRAPHY_ENUM, "tennessen", base='uint8', pos=SIM_OFFSET + 0)
    s = tables.EnumCol(S_ENUM, "NEUTRAL", base='uint8', pos=SIM_OFFSET + 1)
    h = tables.EnumCol(H_ENUM, "0.5", base='uint8', pos=SIM_OFFSET + 2)


class GeneSelectionMixin(tables.IsDescription):
    func = tables.EnumCol(FUNC_ENUM, "LOF_probably", base='uint8', pos=GENE_OFFSET + 0)
    geneset = tables.EnumCol(GENESET_ENUM, "all", base='uint8', pos=GENE_OFFSET + 1)
    min_L = tables.Float64Col(pos=GENE_OFFSET + 2)
    max_L = tables.Float64Col(pos=GENE_OFFSET + 3)


class SimulationHeatmapFixedLength(SimulationHeatmapBase):
    L = tables.Float64Col(pos=2)


class SimulationHeatmapVariableLength(SimulationHeatmapBase, GeneSelectionMixin):
    pass


class EmpiricalHeatmap(HeatmapBase, GeneSelectionMixin):
    pass


def make_heatmap_single_sim(likelihood, ref, sim, s, h, L):
    table = tables_file.get_node(f"/simulated_single/{likelihood}/{ref}/{sim}/{s}/{h}/{L}")
    return heatmap_figure(data_group)


def make_heatmap_geneset_sim(likelihood, ref, sim, s, h, func, geneset, min_L, max_L):
    data_group = tables_file.get_node(f"/simulated_geneset/{likelihood}/{ref}/{sim}/{s}/{h}/{func}/{geneset}/{min_L}/{max_L}")
    return heatmap_figure(data_group)


def make_heatmap_empirical(likelihood, demography, func, genelist, min_L, max_L):
    data_group = tables_file.get_node(f"/exac/{likelihood}/{demography}/{func}/{genelist}/{min_L}/{max_L}")
    return heatmap_figure(data_group)