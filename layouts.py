import base64
import re
import json
import os.path

import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import numpy as np
from dash import callback_context, no_update
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate

from data import make_heatmap_single_sim, make_heatmap_geneset_sim, \
    make_plot_empirical

S_VALUES = ['NEUTRAL', '-4.0', '-3.0', '-2.0', '-1.0']
H_VALUES = ['0.0', '0.1', '0.3', '0.5']


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

    def attach_to_app(self, app):
        app.layout = self.render_layout
        self.register_callbacks(app)


class GeneSelectControls(DashLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geneset_dropdown = self.make_component(dcc.Dropdown, "geneset-dropdown",
                                                    options=[{'label': 'Upload custom list', 'value': 'custom'},
                                                             {'label': 'ConsangBP', 'value': 'inbred_ALL'},
                                                             {'label': 'HI80', 'value': 'haplo_Hurles_80'},
                                                             {'label': 'HI20', 'value': 'haplo_Hurles_low20'},
                                                             {'label': 'CGD AR', 'value': 'CGD_AR_2020'},
                                                             {'label': 'CGD AD', 'value': 'CGD_AD_2020'},
                                                             {'label': 'Lethal AR', 'value': 'Molly_recessive_lethal'}],
                                                    placeholder="Select a gene list")
        self.quality_dropdown = self.make_component(dcc.Dropdown, "quality-dropdown",
                                                    options=[{'label': 'ClinVar High Quality', 'value': 'high'},
                                                             {'label': 'ClinVar Low Quality', 'value': 'low'}],
                                                    placeholder="Select a quality filter")
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
                     options=[{'label': "LOF + damaging missense", 'value': 'LOF_probably'},
                              {'label': "synonymous", 'value': 'synon'}],
                     value="LOF_probably", clearable=False)
        self.length_slider = self.make_component(dcc.RangeSlider, "L-slider", min=0, max=6, step=0.1,
                    marks={0: '10⁰', 1: '10¹', 2: '10²', 3: '10³', 4: '10⁴', 5: '10⁵', 6: '10⁶'},
                    value=[2.5, 5],
                    tooltip={'always_visible': False})

        self.tag_callback(self.switch_custom_selection_visibility,
                          Output(self.custom_select_div.id, "hidden"),
                          [Input(self.geneset_dropdown.id, "value")])
        self.tag_callback(self.update_custom_genes,
                          [Output(self.genes_store.id, "data"),
                           Output(self.genes_textbox_label.id, "children"),
                           Output(self.genes_upload_label.id, "children")],
                          [Input(self.genes_update_button.id, "n_clicks"),
                           Input(self.genes_upload.id, "contents"),
                           Input(self.genes_upload.id, "filename")],
                          [State(self.genes_textbox.id, "value")])

    def render_gene_select_sublayout(self):
        return [
            self.geneset_dropdown,
            self.quality_dropdown,
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
                return no_update, no_update, f"Error processing file {upload_filename}"
        else:
            raise PreventUpdate
        genes = re.split(r"[\s,]+", text.upper())
        genes_set = set(genes)
        return (genes,
               f"loaded {len(genes_set)} unique genes" if source == "textbox" else [],
               f"loaded {len(genes_set)} unique genes from {upload_filename}" if source == "upload" else [])

    def render_layout(self):
        return self.render_gene_select_sublayout()

sim_caption_template = "<b>{plot_type}</b> of maximum likelihood values observed in each of the 17 selection and dominance classes, " \
                   "shown for <b>simulated</b> genes with s=<b>{s}</b>, h=<b>{h}</b>, and length <b>{length}</b>."

class SimsTab(GeneSelectControls):
    def __init__(self):
        super().__init__(id_suffix="-sim")
        self.heatmap = self.make_component(dcc.Graph, 'heatmap')
        self.caption = self.make_component(html.Label, 'caption', sim_caption_template.format(plot_type="histogram",
                                                                                              s="0", h="N/A", length="10<sup>3</sup> sites"))
        self.h_slider = self.make_component(dcc.Slider, "h-slider", min=0, max=3,
                           marks={0: '0.0', 1: '0.1', 2: '0.3', 3: '0.5'},
                           disabled=True, value=3)
        self.s_slider = self.make_component(dcc.Slider, "s-slider", min=0, max=4,
                           marks={0: 'Neutral', 1: '-10⁻⁴', 2: '-10⁻³', 3: '-10⁻²', 4: '-10⁻¹'},
                           value=0)
        self.length_select_mode = self.make_component(dcc.RadioItems, "L-select-mode",
                                                      options=[{'label': "Empirical length profile",
                                                                'value': 'empirical'},
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
                           Input(self.quality_dropdown.id, 'value'),
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
                            self.caption,
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
            html.Div(className="loader-wrapper",
                     children=[dcc.Loading(self.heatmap,
                                              type="circle",
                                              style={'margin-left': "60%"})],
                     style={'width': '60%',
                            'display': 'inline-block',
                           'float': 'right'})],
            style={'width': '800px'})

    def update_heatmap(self, h_idx, s_idx, func, geneset, quality, L_boundaries, single_L, L_mode, custom_genelist):
        if s_idx == 0:
            h_idx = 3
        if L_mode == "single":
            return make_heatmap_single_sim("prf", "supertennessen", "supertennessen", S_VALUES[s_idx],
                                           H_VALUES[h_idx], single_L)
        elif L_mode == "empirical":
            L_boundaries = np.clip(L_boundaries, 2.0, 5.0)
            if geneset == "custom":
                if custom_genelist:
                    geneset = frozenset(custom_genelist)
                else:
                    raise PreventUpdate
            return make_heatmap_geneset_sim("prf", "supertennessen", "supertennessen", S_VALUES[s_idx], H_VALUES[h_idx],
                                            func, geneset, quality, L_boundaries[0], L_boundaries[1])
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
        self.heatmap_mode_switch = self.make_component(daq.ToggleSwitch, "heatmap-switch", value=True)
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
                           Input(self.quality_dropdown.id, 'value'),
                           Input(self.length_slider.id, 'value'),
                           Input(self.color_scheme_buttons.id, 'value'),
                           Input(self.heatmap_mode_switch.id, 'value'),
                           Input(self.genes_store.id, 'data')])
        self.tag_callback(self.enable_disable_color_select,
                          [Output(self.color_scheme_buttons.id, "options"),
                           Output(self.color_scheme_buttons.id, "value")],
                          [Input(self.geneset_dropdown.id, "value"),
                           Input(self.quality_dropdown.id, "value")],
                          [State(self.color_scheme_buttons.id, "options")])
        self.tag_callback(self.do_toggle,
                          Output(self.heatmap_mode_switch.id, "label"),
                          [Input(self.heatmap_mode_switch.id, "value")])

    def render_layout(self):
        return html.Div([html.Div([html.Label("Values to Plot"),
                                self.color_scheme_buttons] +
                                self.render_gene_select_sublayout() +
                                [self.heatmap_mode_switch],
                                style={'width': '30%',
                                        'margin-left': '5%',
                                        'margin-right': '5%',
                                        'margin-top': '10%',
                                        'display': 'inline-block'}),
                        html.Div(className="loader-wrapper",
                                 children=[dcc.Loading(self.heatmap,
                                                          type="circle",
                                                          style={'margin-left': "60%"})],
                                style={'width': '60%',
                                        'display': 'inline-block',
                                        'float': 'right'})],
                style={'width': '800px'})

    def update_heatmap(self, func, geneset, quality, Ls, z_variable, heatmap_mode, custom_genelist):
        if geneset == "custom":
            if custom_genelist:
                geneset = frozenset(custom_genelist)
            else:
                raise PreventUpdate
        return make_plot_empirical("prf", "supertennessen", func, geneset, quality, Ls[0], Ls[1], z_variable, heatmap_mode)

    def enable_disable_color_select(self, geneset, quality, options):
        if geneset is None and quality is None:
            for option in options:
                if option["value"] != "histogram":
                    option["disabled"] = True
            return options, "histogram"
        else:
            for option in options:
                option["disabled"] = False
            return options, no_update

    def do_toggle(self, switch_state):
        if switch_state:
            return "Showing all selection classes"
        else:
            return "Showing only strong selection"

class TwoTabLayout(DashLayout):
    def __init__(self):
        super().__init__()
        self.sims_tab = SimsTab()
        self.exac_tab = ExacTab()
        self.tabs = self.make_component(dcc.Tabs, "tabs", value="sims",
                children=[
                    dcc.Tab(label="Simulated Genes", value='sims',
                            children=self.sims_tab.render_layout()),
                    dcc.Tab(label="ExAC Genes", value='exac',
                            children=self.exac_tab.render_layout())])

        self.tag_callback(self.transfer_gene_select_params_callback("sims"),
                          [Output(self.sims_tab.func_dropdown.id, "value"),
                           Output(self.sims_tab.geneset_dropdown.id, "value"),
                           Output(self.sims_tab.quality_dropdown.id, "value"),
                           Output(self.sims_tab.length_slider.id, "value"),
                           Output(self.sims_tab.genes_textbox.id, "value"),
                           Output(self.sims_tab.genes_update_button.id, "n_clicks"),
                           Output(self.sims_tab.genes_upload.id, "contents"),
                           Output(self.sims_tab.genes_upload.id, "filename"),
                           Output(self.sims_tab.heatmap.id, "loading_state")],
                          [Input(self.tabs.id, "value")],
                          [State(self.exac_tab.func_dropdown.id, "value"),
                           State(self.exac_tab.geneset_dropdown.id, "value"),
                           State(self.exac_tab.quality_dropdown.id, "value"),
                           State(self.exac_tab.length_slider.id, "value"),
                           State(self.exac_tab.genes_textbox.id, "value"),
                           State(self.sims_tab.genes_update_button.id, "n_clicks"),
                           State(self.exac_tab.genes_upload.id, "contents"),
                           State(self.exac_tab.genes_upload.id, "filename"),
                           State(self.exac_tab.genes_textbox_label.id, "children"),
                           State(self.exac_tab.genes_upload_label.id, "children")])

        self.tag_callback(self.transfer_gene_select_params_callback("exac"),
                          [Output(self.exac_tab.func_dropdown.id, "value"),
                           Output(self.exac_tab.geneset_dropdown.id, "value"),
                           Output(self.exac_tab.quality_dropdown.id, "value"),
                           Output(self.exac_tab.length_slider.id, "value"),
                           Output(self.exac_tab.genes_textbox.id, "value"),
                           Output(self.exac_tab.genes_update_button.id, "n_clicks"),
                           Output(self.exac_tab.genes_upload.id, "contents"),
                           Output(self.exac_tab.genes_upload.id, "filename"),
                           Output(self.exac_tab.heatmap.id, "loading_state")],
                          [Input(self.tabs.id, "value")],
                          [State(self.sims_tab.func_dropdown.id, "value"),
                           State(self.sims_tab.geneset_dropdown.id, "value"),
                           State(self.sims_tab.quality_dropdown.id, "value"),
                           State(self.sims_tab.length_slider.id, "value"),
                           State(self.sims_tab.genes_textbox.id, "value"),
                           State(self.exac_tab.genes_update_button.id, "n_clicks"),
                           State(self.sims_tab.genes_upload.id, "contents"),
                           State(self.sims_tab.genes_upload.id, "filename"),
                           State(self.sims_tab.genes_textbox_label.id, "children"),
                           State(self.sims_tab.genes_upload_label.id, "children")])



    def render_layout(self):
        return self.tabs

    def register_callbacks(self, app):
        super().register_callbacks(app)
        self.sims_tab.register_callbacks(app)
        self.exac_tab.register_callbacks(app)


    @staticmethod
    def transfer_gene_select_params_callback(target):
        def transfer_gene_select_params(tab, func, geneset, quality, L_boundaries,
                                        box_text, button_clicks,
                                        upload_data, upload_filename,
                                        box_label, upload_label):
            if tab != target:
                raise PreventUpdate
            return (func, geneset, quality, L_boundaries,
                    box_text if box_label else no_update,
                    button_clicks + 1 if box_label else no_update,
                    upload_data if upload_label else no_update,
                    upload_filename if upload_label else no_update,
                    {"is_loading": True} )
        return transfer_gene_select_params

