"""
This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.

dcc.Location is used to track the current location, and a callback uses the
current location to render the appropriate page content. The active prop of
each NavLink is set automatically according to the current pathname. To use
this feature you must install dash-bootstrap-components >= 0.11.0.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""
import dash
import os
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
import pandas as pd
import plotly.express as px

# app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app = dash.Dash(external_stylesheets=[dbc.themes.COSMO])

server = app.server

models_cat = pd.read_csv('./pm_bias_models_fin.csv')
pm_bias = pd.read_csv('./pm_bias_fin.csv')
polesAB = pd.read_csv('./polesAB.csv')

height_width, height_width_subp = 700, 600

fig_l2 = px.sunburst(models_cat, path=['Model','Craton'], values='biasp', color_discrete_sequence=["darkred","darkred","darkred","darkred","darkred","darkred"],width=height_width, height=height_width) # , hover_data=['FROMAGE','TOAGE', 'RecEMinAge','RecEMin','poleA95']
fig_l2.update_traces(opacity=0.8)
fig_l2.update_layout(uniformtext=dict(minsize=9))

fig_l5 = px.sunburst(models_cat, path=['Model','Q4YN','Category','Craton','GPMDB_result#'], values='biasp', width=800, height=800)#,color_discrete_sequence=["lightblue","lightcoral","lightyellow","lightsteelblue","lightgreen"]) , hover_data=['FROMAGE','TOAGE', 'RecEMinAge','RecEMin','poleA95']
# fig_l5.update_traces(opacity=0.6)
fig_l5.update_layout(uniformtext=dict(minsize=9))

fig_age = px.bar(models_cat, x="RecEMinAge", y="biasp", color="Craton",  barmode="group", facet_col="Model", facet_row='Category',hover_data=['GPMDB_result#','FROMAGE','TOAGE', 'RecEMin','poleA95','Description'], color_discrete_sequence=["black","blue", "green","salmon", "orange", "mediumpurple"], height=600)
fig_age.update_traces(width=3, opacity=0.5)#, marker_color=['rgb(255,0,0)','rgb(255,0,0)', 'rgb(255,0,0)', 'rgb(255,0,0)']
fig_age.update_layout(template="simple_white", legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1))
fig_age.update_xaxes(showgrid=False, gridwidth=1, mirror=True)
fig_age.update_yaxes(range=[0, 90],showgrid=True, gridwidth=1, mirror=True)

fig_m21 = px.sunburst(models_cat[models_cat.Model=='M21'], path=['Model','Craton','Q4YN','Category','GPMDB_result#'], values='biasp', hover_data=['FROMAGE','TOAGE', 'RecEMinAge','RecEMin','poleA95'],color_discrete_sequence=["green","blue", "orange", "mediumpurple", "salmon"], height=height_width_subp)
fig_m21.update_layout(uniformtext=dict(minsize=9))
fig_m21.update_traces(opacity=0.6)

fig_m17 = px.sunburst(models_cat[models_cat.Model=='M17'], path=['Model','Craton','Q4YN','Category','GPMDB_result#'], values='biasp', hover_data=['FROMAGE','TOAGE', 'RecEMinAge','RecEMin','poleA95'],color_discrete_sequence=["green","blue",  "mediumpurple", "orange","salmon"], height=height_width_subp)
fig_m17.update_layout(uniformtext=dict(minsize=9))
fig_m17.update_traces(opacity=0.6)

fig_r21 = px.sunburst(models_cat[models_cat.Model=='R21'], path=['Model','Craton','Q4YN','Category','GPMDB_result#'], values='biasp', hover_data=['FROMAGE','TOAGE', 'RecEMinAge','RecEMin','poleA95'],color_discrete_sequence=["blue", "green","orange","salmon", "mediumpurple"], height=height_width_subp)
fig_r21.update_layout(uniformtext=dict(minsize=9))
fig_r21.update_traces(opacity=0.6)

fig_l08 = px.sunburst(models_cat[models_cat.Model=='L08'], path=['Model','Craton','Q4YN','Category','GPMDB_result#'], values='biasp', hover_data=['FROMAGE','TOAGE', 'RecEMinAge','RecEMin','poleA95'],color_discrete_sequence=["blue", "green","orange","salmon", "mediumpurple"], height=height_width_subp)
fig_l08.update_layout(uniformtext=dict(minsize=9))
fig_l08.update_traces(opacity=0.6)

fig_tc16 = px.sunburst(models_cat[models_cat.Model=='TC16'], path=['Model','Craton','Q4YN','Category','GPMDB_result#'], values='biasp', hover_data=['FROMAGE','TOAGE', 'RecEMinAge','RecEMin','poleA95'],color_discrete_sequence=["blue", "green","mediumpurple", "orange","salmon"], height=height_width_subp)
fig_tc16.update_layout(uniformtext=dict(minsize=9))
fig_tc16.update_traces(opacity=0.6)

fig_s21 = px.sunburst(models_cat[models_cat.Model=='S21'], path=['Model','Craton','Q4YN','Category','GPMDB_result#'], values='biasp', hover_data=['FROMAGE','TOAGE', 'RecEMinAge','RecEMin','poleA95'],color_discrete_sequence=["blue", "green","mediumpurple","salmon", "orange"], height=height_width_subp)
fig_s21.update_layout(uniformtext=dict(minsize=9))
fig_s21.update_traces(opacity=0.6)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        # html.H2("Sidebar", className="display-4"),
        html.P(
            "Paleomagnetic bias of paleogeographic models", className="lead"
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Summary", href="/", active="exact"),
                dbc.NavLink("Distribution of the paleomagnetic bias", href="/page-1", active="exact"),
                dbc.NavLink("Distribution of the paleomagnetic bias continued ... ...", href="/page-1a", active="exact"),
                dbc.NavLink("Paleomagnetic bias (normalized)", href="/page-2", active="exact"),
                dbc.NavLink("Paleomagnetic Data", href="/page-pmagdata", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.P(
            "Please cite this work: Wu, L., Murphy, J.B., Nance, R.D., Evaluation of paleomagnetic bias in Ediacaran global paleogeographic reconstructions. Under Review.", className="lead"
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

markdown_text_sum0 = '''#### Caption
Interactive nested piecharts showing hierarchy of paleomagnetic bias by continents in individual paleogeographic models. 
Clicking on an individual model for details of paleomagnetic bias for that model. To go back one layer, click in the centre.
Width of each slice is proportional to [paleomagnetic bias](/page-2). 
Models: R21 – Robert et al. (2021); M21 – Merdith et al. (2021); M17 – Merdith et al. (2017); L08 – Li et al. (2008); TC16 - Torsvik and Cocks, 2016; S21 – Scotese et al. (2021).
'''

markdown_text_sum = '''#### Caption
Interactive nested piecharts showing hierarchy of paleomagnetic bias by continents in individual paleogeographic models. Clicking on an individual model for details of paleomagnetic bias for that model. 
Clicking on a craton or data category provides details of paleomagnetic bias for that craton or data category, respectively. To go back one layer, click in the centre.
Digits (outermost arcs) are result numbers of paleopoles from Evans et al. (2021) and the width of each slice is proportional to the paleomagnetic bias with respect to each paleopole. 
Refer to the [Paleomagnetic Data](/page-pmagdata) for detailed information of the paleopoles. 
Models: R21 – Robert et al. (2021); M21 – Merdith et al. (2021); M17 – Merdith et al. (2017); L08 – Li et al. (2008); TC16 - Torsvik and Cocks, 2016; S21 – Scotese et al. (2021).
'''

markdown_text_temp ='''#### Caption
Temporal distribution of the paleomagnetic bias. Refer to the [Paleomagnetic Data](/page-pmagdata) for detailed information of the paleopoles.
Models: R21 – Robert et al. (2021); M21 – Merdith et al. (2021); M17 – Merdith et al. (2017); L08 – Li et al. (2008); TC16 - Torsvik and Cocks, 2016; S21 – Scotese et al. (2021).'''

markdown_text_evans21 ='''Data Source: [Evans et al. (2021)](https://www.sciencedirect.com/science/article/pii/B9780128185339000072), 
An expanding list of reliable paleomagnetic poles for Precambrian tectonic reconstructions, in Pesonen, L.J., Salminen, J., Elming, S.A., Evans, D.A. and Veikkolainen, T. ed., 
Ancient Supercontinents and the Paleogeography of Earth, Elsevier, p. 605–639. '''

# html.Div([ html.H2("Sidebar", className="display-4"), html.Hr(), html.P("Evaluation of paleomagnetic bias in Ediacaran global paleogeographic reconstructions", className="lead") ]), 

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.Div([ html.H3(children='Paleomagnetic bias of Ediacaran paleogeographic models'), 
        html.Div([ html.Div(children=[dcc.Graph(id='pie-2-levels', figure=fig_l2)]), 
        html.Div(children=[dcc.Markdown(children=markdown_text_sum0)]),
        html.Div(children=dcc.Markdown(children='Researchers may use our method [(source code here)](https://github.com/leiwuGeo/pmag_bias) to perform their own objective tests of paleogeographic models.')) ]) ])#  

    elif pathname == "/page-1":
        return html.Div([ html.H3(children='Break-down of the paleomagnetic bias in the various models'), 
        html.Div(children=[dcc.Graph(id='bar-age', figure=fig_age)]),
        html.Div(children=[dcc.Markdown(children=markdown_text_temp)]),
        html.Div(children=[dcc.Graph(id='pie-5-levels', figure=fig_l5)]),
        html.Div(children=[dcc.Markdown(children=markdown_text_sum)]) ]) # 

    elif pathname == "/page-1a":
        return html.Div([ html.H3(children='Break-down of the paleomagnetic bias in the various models'),
        html.Div([
                dbc.Row(
                    [
                        dbc.Col(html.Div(children=[dcc.Graph(id='pie-r21', figure=fig_r21)])),
                        dbc.Col(html.Div(children=[dcc.Graph(id='pie-m21', figure=fig_m21)])),
                        dbc.Col(html.Div(children=[dcc.Graph(id='pie-m17', figure=fig_m17)])),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(children=[dcc.Graph(id='pie-l08', figure=fig_l08)])),
                        dbc.Col(html.Div(children=[dcc.Graph(id='pie-s21', figure=fig_s21)])),
                        dbc.Col(html.Div(children=[dcc.Graph(id='pie-tc16', figure=fig_tc16)])),
                    ]
                ) ]),
        html.Div(children=[dcc.Markdown(children=markdown_text_sum)]) ]) # 

    elif pathname == "/page-2":
        table = dbc.Table.from_dataframe(pm_bias, striped=True, bordered=True, hover=True)
        return html.Div([ html.H3(children='Normalized paleomagnetic bias (measured in degrees) in the global paleogeographic models'),
        html.Div(children=[table]) ]) #    

    elif pathname == "/page-pmagdata":
        table = dbc.Table.from_dataframe(polesAB, striped=True, bordered=True, hover=True)
        return html.Div([ html.H3(children='Ediacaran paleomagnetic data'),
        html.Div(children=[table]), 
        html.Div(children=[dcc.Markdown(children=markdown_text_evans21)]) ]) #   

    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
    app.run_server()
    # app.run_server(debug=True)