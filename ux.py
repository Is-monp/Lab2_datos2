import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# Estilos CSS personalizados
styles = {
    'map-container': {
        'background-color': '#F4F6F8',
        'height': '80vh',
        'flex': '1',
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center'
    },
    'tabs': {
        'background-color': '#2196F3',  # Azul
        'color': 'white',
        'border': 'none',
        'width': '150px',
        'margin-right': '10px'
    },
    'tabs-content': {
        'flex': '3',
        'padding': '20px',
        'background-color': '#FFFFFF',
        'border-radius': '5px',
        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
    }
}

# Layout de la aplicación Dash
app.layout = html.Div([
    html.H1('My map', style={'color': '#2196F3'}),  # Azul

    html.Div([
        html.Div(className='row',id='map-container', style=styles['map-container'], children=[
            html.Img(src='https://via.placeholder.com/800x400', style={'max-width': '100%', 'max-height': '100%'})
        ]),

        # Panel de pestañas en columna
        html.Div([
            dcc.Tabs(id='tabs', value='tab-1', children=[
                dcc.Tab(label='Content', value='tab-1', style=styles['tabs']),
                dcc.Tab(label='Pestaña 2', value='tab-2', style=styles['tabs']),
                dcc.Tab(label='Pestaña 3', value='tab-3', style=styles['tabs']),
            ], vertical=True)
        ],className='row', style={'flex': '1', 'display': 'flex', 'flex-direction': 'column'}),
        
        # Contenido de las pestañas
        html.Div(id='tabs-content', style=styles['tabs-content'])
    ], style={'display': 'flex', 'flex-direction': 'column', 'height': '100vh', 'padding': '20px'})  # Estilo para flexbox y padding
])

# Callback para renderizar el contenido de las pestañas
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Contenido de la Pestaña 1', style={'color': '#2196F3'}),  # Azul
            html.P('Este es el contenido de la primera pestaña.', style={'color': '#333'})
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Contenido de la Pestaña 2', style={'color': '#2196F3'}),  # Azul
            html.P('Este es el contenido de la segunda pestaña.', style={'color': '#333'})
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Contenido de la Pestaña 3', style={'color': '#2196F3'}),  # Azul
            html.P('Este es el contenido de la tercera pestaña.', style={'color': '#333'})
        ])

# Ejecutar la aplicación Dash
if __name__ == '__main__':
    app.run_server(debug=True)



