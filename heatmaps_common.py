import base64
import os
import re
import warnings
from functools import lru_cache, wraps
from pathlib import Path, PurePosixPath
from typing import Tuple, List, Any, Dict
from urllib.parse import quote

import flask
import numpy as np
import pandas as pd
from dash import callback_context, no_update
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import mpmath
import attr

BASE_DIR = os.path.join(os.path.dirname(__file__), "dominance_data")
SIM_DATA_TEMPLATE = os.path.join(BASE_DIR, "sims",
                                 "{likelihood}_ref_{ref}_sims_{sim}_S_{s}_h_{h}_L_{L:.1f}.tsv")
EXAC_SUMSTATS_PATH = os.path.join(BASE_DIR, "ExAC_63K_symbol_plus_ensembl_func_summary_stats.tsv")

LIKELIHOOD_FILE = "ExAC_63K_prf_supertennessen_inference.tsv"

LIKELIHOODS = ['prf']
DEMOGRAPHIES = ['supertennessen']
S_VALUES = ['NEUTRAL', '-4.0', '-3.0', '-2.0', '-1.0']
H_VALUES = ['0.0', '0.1', '0.3', '0.5']
FUNCS = ['LOF_probably', 'synon']
GENESETS = ['all', 'haplo_Hurles_80', 'CGD_AD_2020', 'inbred_ALL', 'haplo_Hurles_low20', 'CGD_AR_2020']

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


def filter_df(df, func, genelist, min_L, max_L):
    selector = df.func == func
    selector &= df.U.between(10**(min_L-8), 10**(max_L-8))
    if genelist != "all":
        if isinstance(genelist, (set, frozenset)):
            selector &= df.gene.isin(genelist)
        else:
            selector &= df.gene.isin(GENESETS_DICT[genelist])
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
def load_exac_data(likelihood, demography, func, geneset, min_L, max_L):
    geneset_df = load_filtered_df(demography, func, geneset, likelihood, min_L, max_L)
    geneset_histogram = np.array(extract_histogram_empirical(geneset_df), dtype=float)
    geneset_count = len(geneset_df)
    if geneset == "all":
        ones = np.ones_like(geneset_histogram)
        ones[np.isnan(geneset_histogram)] = np.nan
        return geneset_histogram, ones, ones
    all_df = load_filtered_df(demography, func, "all", likelihood, min_L, max_L)
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


def load_filtered_df(demography, func, genelist, likelihood, min_L, max_L):
    unfiltered_df = load_unfiltered_df(likelihood, demography)
    filtered_df = filter_df(unfiltered_df, func, genelist, min_L, max_L)
    return filtered_df


def create_app(app_name, app_filename):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app_path = Path(app_filename)
    app_url = PurePosixPath('/', app_path.parent.name, app_path.name)
    server = flask.Flask(app_name)
    server.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app = dash.Dash(server=server, external_stylesheets=external_stylesheets,
                    requests_pathname_prefix=quote(str(app_url) + "/"))
    if "dev" in __file__:
        app.enable_dev_tools(debug=True)
    return app


class DashLayout:
    def __init__(self, id_suffix=""):
        self.id_suffix = id_suffix
        self.callbacks = []
        self.callbacks_registered = False

    def make_component(self, factory, id, *args, **kwargs):
        return factory(id=f"{id}{self.id_suffix}", *args, **kwargs)

    def render_layout(self):
        return []

    def tag_callback(self, callback_method, *args, **kwargs):
        self.callbacks.append((callback_method, args, kwargs))

    def register_callbacks(self, app):
        if not self.callbacks_registered:
            for callback_method, args, kwargs in self.callbacks:
                app.callback(*args, **kwargs)(callback_method)
            self.callbacks_registered = True


class GeneSelectControls(DashLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geneset_dropdown = self.make_component(dcc.Dropdown, "geneset-dropdown",
                                                    options=[{'label': 'All genes', 'value': 'all'},
                                                             {'label': 'HI > 80%', 'value': 'haplo_Hurles_80'},
                                                             {'label': 'CGD AD', 'value': 'CGD_AD_2020'},
                                                             {'label': 'Inbred', 'value': 'inbred_ALL'},
                                                             {'label': 'HI < 20%', 'value': 'haplo_Hurles_low20'},
                                                             {'label': 'CGD AR', 'value': 'CGD_AR_2020'},
                                                             {'label': 'Custom list', 'value': 'custom'}],
                                                    value='all', style={'width': '7em', 'display': 'inline-block'})
        self.genes_textbox = self.make_component(dcc.Textarea, "genes-textbox")
        self.genes_update_button = self.make_component(html.Button, "update-button", "Update", n_clicks=0)
        self.genes_textbox_label = self.make_component(html.Label, "textbox-genes-label")
        self.genes_upload = self.make_component(dcc.Upload, 'genes-upload', children=[html.Button("Select file")])
        self.genes_upload_label = self.make_component(html.Label, "upload-genes-label")
        self.genes_store = self.make_component(dcc.Store, "custom-genes")
        self.custom_select_div = self.make_component(html.Div, "custom-select",
            [
                html.Label("Enter gene symbols, one per line or separated by commas or spaces, "
                           "or upload a file with gene symbols."),
                self.genes_textbox, self.genes_textbox_label, self.genes_update_button,
                self.genes_upload, self.genes_upload_label,
                self.genes_store
            ])
        self.func_dropdown = self.make_component(dcc.Dropdown, "func-dropdown",
                     options=[{'label': "LOF + PolyPhen probably", 'value': 'LOF_probably'},
                              {'label': "synonymous", 'value': 'synon'}],
                     value="LOF_probably", style={'width': '15em', 'display': 'inline-block'})
        self.length_slider = self.make_component(dcc.RangeSlider, "L-slider", min=0, max=6, step=0.1,
                    marks={0: '10⁰', 1: '10¹', 2: '10²', 3: '10³', 4: '10⁴', 5: '10⁵', 6: '10⁶'},
                    value=[2, 5],
                    tooltip={'always_visible': False})

        self.tag_callback(self.switch_custom_selection_visibility,
                          Output(self.custom_select_div.id, "hidden"),
                          [Input(self.geneset_dropdown.id, "value")])
        self.tag_callback(self.update_custom_genes,
                          [Output(self.genes_store.id, "data"),
                           Output(self.genes_textbox_label.id, "children"),
                           Output(self.genes_upload_label.id, "children"),
                           Output(self.genes_textbox.id, "value")],
                          [Input(self.genes_update_button.id, "n_clicks"),
                           Input(self.genes_upload.id, "contents"),
                           Input(self.genes_upload.id, "filename")],
                          [State(self.genes_textbox.id, "value")])

    def render_gene_select_sublayout(self):
        return [
            self.geneset_dropdown,
            self.custom_select_div,
            self.func_dropdown,
            html.Label("L"), self.length_slider
        ]

    @staticmethod
    def switch_custom_selection_visibility(geneset):
        return geneset != "custom"

    def update_custom_genes(self, button_clicks, upload_data, upload_filename, textbox_value):
        triggered_prop = callback_context.triggered[0]['prop_id']
        if triggered_prop == f"{self.genes_update_button.id}.n_clicks":
            source = "textbox"
            text = textbox_value
        elif triggered_prop == f"{self.genes_upload.id}.contents":
            source = "upload"
            try:
                content_type, encoded_content = upload_data.split(",")
                text = base64.b64decode(encoded_content).decode('utf8')
            except ValueError:
                return no_update, no_update, f"Error processing file {upload_filename}", no_update
        else:
            raise PreventUpdate
        genes = re.split(r"[\s,]+", text.upper())
        genes_set = set(genes)
        return (genes,
               f"parsed {len(genes_set)} unique genes" if source == "textbox" else [],
               f"loaded {len(genes_set)} unique genes from {upload_filename}" if source == "upload" else [],
               "\n".join(genes_set) if source == "textbox" else no_update)

    def render_layout(self):
        return self.render_gene_select_sublayout()


class SimsTab(GeneSelectControls):
    h_labels = ["0.0", "0.1", "0.3", "0.5"]
    s_labels = ["NEUTRAL", "-4.0", "-3.0", "-2.0", "-1.0"]

    def __init__(self):
        super().__init__(id_suffix="-sim")
        self.heatmap = self.make_component(dcc.Graph, 'heatmap')
        self.h_slider = self.make_component(dcc.Slider, "h-slider", min=0, max=3,
                           marks={0: '0.0', 1: '0.1', 2: '0.3', 3: '0.5'},
                           disabled=True, value=3)
        self.s_slider = self.make_component(dcc.Slider, "s-slider", min=0, max=4,
                           marks={0: 'Neutral', 1: '-10⁻⁴', 2: '-10⁻³', 3: '-10⁻²', 4: '-10⁻¹'},
                           value=0)
        self.length_select_mode = self.make_component(dcc.RadioItems, "L-select-mode",
                                                      options=[{'label': "Empirical length profile", 'value': 'empirical'},
                                        {'label': "Single length", 'value': 'single'}], value='single')
        self.length_slider_single = self.make_component(dcc.Slider, "L-slider-single",
                                                        min=2, max=5, step=0.1,
                                                        marks={2: '10²', 3: '10³', 4: '10⁴', 5: '10⁵'},
                                                        value=3,
                                                        tooltip={'always_visible': False})
        self.length_select_single_div = self.make_component(html.Div, "L-select-single",
                                                            [html.Label("L"), self.length_slider_single])
        self.length_select_empirical_div = self.make_component(html.Div, "L-select-empirical",
                                                               self.render_gene_select_sublayout())

        self.tag_callback(self.update_heatmap,
                          Output(self.heatmap.id, 'figure'),
                          [Input(self.h_slider.id, 'value'),
                           Input(self.s_slider.id, 'value'),
                           Input(self.func_dropdown.id, 'value'),
                           Input(self.geneset_dropdown.id, 'value'),
                           Input(self.length_slider.id, 'value'),
                           Input(self.length_slider_single.id, 'value'),
                           Input(self.length_select_mode.id, 'value'),
                           Input(self.genes_store.id, 'data')])
        self.tag_callback(self.enable_disable_h_slider,
                          Output(self.h_slider.id, 'disabled'),
                          [Input(self.s_slider.id, 'value')])
        self.tag_callback(self.switch_L_selection_visibility,
                          [Output(self.length_select_single_div.id, 'hidden'),
                           Output(self.length_select_empirical_div.id, 'hidden')],
                          [Input(self.length_select_mode.id, 'value')]
                          )

    def render_layout(self):
        return html.Div(children=[
                        html.Div(children=[
                            html.Label("h"), self.h_slider,
                            html.Br(),
                            html.Label("s"), self.s_slider,
                            html.Br(),
                            self.length_select_mode,
                            self.length_select_single_div,
                            self.length_select_empirical_div
                        ], style={'width': '30%',
                                  'margin-left': '5%',
                                  'margin-right': '5%',
                                  'margin-top': '10%',
                                  'display': 'inline-block'}),
            html.Div(children=[self.heatmap],
                     style={'width': '60%',
                      'display': 'inline-block',
                      'float': 'right'})],
            style={'width': '800px'})

    def update_heatmap(self, h_idx, s_idx, func, geneset, L_boundaries, single_L, L_mode, custom_genelist):
        if s_idx == 0:
            h_idx = 3
        if L_mode == "single":
            return make_heatmap_single_sim("prf", "supertennessen", "supertennessen", self.s_labels[s_idx],
                                           self.h_labels[h_idx], single_L)
        elif L_mode == "empirical":
            L_boundaries = np.clip(L_boundaries, 2.0, 5.0)
            if geneset == "custom":
                if custom_genelist:
                    geneset = frozenset(custom_genelist)
                else:
                    raise PreventUpdate
            return make_heatmap_geneset_sim("prf", "supertennessen", "supertennessen", self.s_labels[s_idx],
                                            self.h_labels[h_idx], func, geneset, L_boundaries[0], L_boundaries[1])
        else:
            raise ValueError(f"Unknown L selection mode {L_mode}")

    def enable_disable_h_slider(self, s_slider_value):
        return s_slider_value == 0

    def switch_L_selection_visibility(self, mode):
        if mode == "empirical":
            return True, False
        elif mode == "single":
            return False, True
        else:
            raise ValueError(f"Unrecognized L selection mode {mode}")


class ExacTab(GeneSelectControls):
    def __init__(self):
        super().__init__(id_suffix="-exac")
        self.color_scheme_buttons = self.make_component(dcc.RadioItems, "color-buttons",
                       options=[{'label': 'Histogram', 'value': 'histogram'},
                                {'label': 'Enrichment (log odds ratio)', 'value': 'odds_ratio'},
                                {'label': 'Enrichment (p-value)', 'value': 'p_value'}],
                       value='histogram')
        self.heatmap = self.make_component(dcc.Graph, 'heatmap')
        self.tag_callback(self.update_heatmap,
                          Output(self.heatmap.id, 'figure'),
                          [Input(self.func_dropdown.id, 'value'),
                           Input(self.geneset_dropdown.id, 'value'),
                           Input(self.length_slider.id, 'value'),
                           Input(self.color_scheme_buttons.id, 'value'),
                           Input(self.genes_store.id, 'data')])

    def render_layout(self):
        return html.Div([html.Div([html.Label("Color Scheme"),
                                self.color_scheme_buttons] +
                                self.render_gene_select_sublayout(),
                                style={'width': '30%',
                                        'margin-left': '5%',
                                        'margin-right': '5%',
                                        'margin-top': '10%',
                                        'display': 'inline-block'}),
                        html.Div([self.heatmap],
                                style={'width': '60%',
                                        'display': 'inline-block',
                                        'float': 'right'})],
                style={'width': '800px'})

    def update_heatmap(self, func, geneset, Ls, z_variable, custom_genelist):
        if geneset == "custom":
            if custom_genelist:
                geneset = frozenset(custom_genelist)
            else:
                raise PreventUpdate
        return make_heatmap_empirical("prf", "supertennessen", func, geneset, Ls[0], Ls[1], z_variable)


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


def make_heatmap_geneset_sim(likelihood, ref, sim, s, h, func, geneset, min_L, max_L):
    filtered_df = load_filtered_df(ref, func, geneset, likelihood, min_L, max_L)
    L = filtered_df.U.transform('log10') + 8.0
    try:
        histogram = load_sim_data(likelihood, ref, sim, s, h, L)
    except ValueError:
        histogram = get_null_histogram()
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = histogram / np.nansum(histogram)
    return heatmap_figure({'histogram': histogram,
                           "frac": frac})


def make_heatmap_empirical(likelihood, demography, func, genelist, min_L, max_L, z_variable="histogram"):
    try:
        histogram, odds_ratio, p_value = load_exac_data(likelihood, demography, func, genelist, min_L, max_L)
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
