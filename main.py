from typing import Any, List, Optional, Tuple
from pprint import pprint
from queue import Queue
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import pandas as pd
import numpy as np
import graphviz as gv


def read_flight_data(filename):
    df = pd.read_csv(filename)
    df.index = range(1, len(df) + 1)  # Para iniciar el conteo del índice en 1 en lugar de 0
    return df

def convert_to_radians(df):
    df['Latitud_salida_rad'] = np.radians(df['Source Airport Latitude'])
    df['Longitud_salida_rad'] = np.radians(df['Source Airport Longitude'])
    df['Latitud_llegada_rad'] = np.radians(df['Destination Airport Latitude'])
    df['Longitud_llegada_rad'] = np.radians(df['Destination Airport Longitude'])
    return df

def calculate_differences(df):
    df['dLat'] = df['Latitud_llegada_rad'] - df['Latitud_salida_rad']
    df['dLon'] = df['Longitud_llegada_rad'] - df['Longitud_salida_rad']
    return df

def calculate_distance(df, radio_tierra=6371.0):
    df['distancia'] = 2 * np.arcsin(np.sqrt(
        np.sin(df['dLat'] / 2)**2 +
        np.cos(df['Latitud_salida_rad']) * np.cos(df['Latitud_llegada_rad']) *
        np.sin(df['dLon'] / 2)**2
    )) * radio_tierra
    return df

class Grafo:
    def __init__(self):
        self.graf = {}

    def add_edge(self, source, destination, distance):
        if source not in self.graf:
            self.graf[source] = {}
        if destination not in self.graf:
            self.graf[destination] = {}

        self.graf[source][destination] = distance
        self.graf[destination][source] = distance

    def __str__(self):
        result = ""
        for source in self.graf:
            for destination, distance in self.graf[source].items():
                result += f"({source} -- {destination} : {distance} km)\n"
        return result

    def dijkstra(self, source_airport, destination_airport):
        if source_airport not in self.graf or destination_airport not in self.graf:
            return None  # Al menos uno de los aeropuertos no existe en el grafo

        distances = {vertex: float('inf') for vertex in self.graf}
        distances[source_airport] = 0

        previous_vertices = {}  # Para mantener un seguimiento de los vértices anteriores en el camino mínimo
        priority_queue = [(0, source_airport)]

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_vertex == destination_airport:
                break

            if current_distance > distances[current_vertex]:
                continue

            for neighbor, weight in self.graf[current_vertex].items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_vertices[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (distance, neighbor))

        if distances[destination_airport] == float('inf'):
            return None

        path = []
        current = destination_airport
        while current != source_airport:
            path.append(current)
            current = previous_vertices[current]
        path.append(source_airport)

        path.reverse()

        return path

    def dijkstra_distance(self, source_airport, destination_airport):
        if source_airport not in self.graf or destination_airport not in self.graf:
            return None  # Al menos uno de los aeropuertos no existe en el grafo

        distances = {vertex: float('inf') for vertex in self.graf}
        distances[source_airport] = 0

        previous_vertices = {}  # Para mantener un seguimiento de los vértices anteriores en el camino mínimo
        priority_queue = [(0, source_airport)]

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_vertex == destination_airport:
                break

            if current_distance > distances[current_vertex]:
                continue

            for neighbor, weight in self.graf[current_vertex].items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_vertices[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (distance, neighbor))

        if distances[destination_airport] == float('inf'):
            return None

        path = []
        current = destination_airport
        while current != source_airport:
            path.append(current)
            current = previous_vertices[current]
        path.append(source_airport)

        path.reverse()

        total_distance = 0
        for i in range(len(path) - 1):
            current_airport = path[i]
            next_airport = path[i + 1]
            distance_between_airports = self.graf[current_airport][next_airport]
            total_distance += distance_between_airports
        return total_distance

def search_nodes(search, df):
    flights = pd.DataFrame({col: [] for col in df.keys()})

    if len(search) == 0:
        return df

    for i in df.index:
        if all([any([df[col][i] == crit for crit in search[col]]) for col in search.keys()]):
            flight = pd.DataFrame({col: df[col][i] for col in df.keys()}, index=[len(flights) + 1])
            flights = pd.concat([flights, flight])

    return flights

def get_nodes_from_flights(df):
    cols = [col for col in df.keys()]
    nodes_names_set = set()
    nodes_set = []

    for i in df.index:
        if df['Source Airport Name'][i] not in nodes_names_set:
            nodes_names_set.add(df['Source Airport Name'][i])
            nodes_set.append([df[col][i] for col in cols[:6]])

        if df['Destination Airport Name'][i] not in nodes_names_set:
            nodes_names_set.add(df['Destination Airport Name'][i])
            nodes_set.append([df[col][i] for col in cols[6:12]])

    cols = ['Airport Code', 'Airport Name',
            'Airport City', 'Airport Country',
            'Airport Latitude', 'Airport Longitude']

    nodes = pd.DataFrame({col: [] for col in cols})
    for node_set in nodes_set:
        node = pd.DataFrame({cols[i]: node_set[i] for i in range(len(node_set))}, index=[len(nodes) + 1])
        nodes = pd.concat([nodes, node])

    return nodes

def graph_nodes(search, df, draw_lines):
    mapbox_access_token = 'pk.eyJ1Ijoic2ViYXN0aWFubWFsZG9uYWRvMTk0NSIsImEiOiJjbGluYnRobHkwbDQyM2xwOGc4aGN5ZnpvIn0.Jal1X7da0VhVK8gkKrWBng'

    flights = search_nodes(search, df)
    nodes = get_nodes_from_flights(flights)

    lon = [nodes['Airport Longitude'][i] for i in nodes.index]
    lat = [nodes['Airport Latitude'][i] for i in nodes.index]

    fig = go.Figure()

    if draw_lines:
        for i in flights.index:
            fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[flights['Source Airport Longitude'][i], flights['Destination Airport Longitude'][i]],
            lat=[flights['Source Airport Latitude'][i], flights['Destination Airport Latitude'][i]],
            hoverinfo='skip',
            marker={'size': 1}))

    fig.add_trace(go.Scattermapbox(
        mode="markers+text",
        lon=lon, lat=lat,
        marker={'size': 20, 'symbol': ["airport" for i in range(len(nodes))]},
        text=[nodes['Airport Name'][i] for i in nodes.index], textposition="bottom right"))

    fig.update_layout(
        mapbox={
            'accesstoken': mapbox_access_token,
            'style': "outdoors", 'zoom': 0.7},
        showlegend=False)

    fig.show(renderer="browser")

def main():
    filename = "flights.csv"
    df = read_flight_data(filename)
    df = convert_to_radians(df)
    df = calculate_differences(df)
    df = calculate_distance(df)

    flight_graph = Grafo()
    for _, row in df.iterrows():
        source_airport = row['Source Airport Code'].strip().upper()
        destination_airport = row['Destination Airport Code'].strip().upper()
        distance = row['distancia']
        flight_graph.add_edge(source_airport, destination_airport, distance)

    source_airport = input("Ingrese el código del aeropuerto de origen: ").strip().upper()
    destination_airport = input("Ingrese el código del aeropuerto de destino: ").strip().upper()
    camino_minimo = flight_graph.dijkstra(source_airport, destination_airport)

    if camino_minimo:
        print("Camino mínimo:")
        print(" -> ".join(camino_minimo))
    else:
        print("No se encontró un camino entre los aeropuertos especificados.")

    graph_nodes({'Source Airport Code': ['CAN']}, df, True)

    source = input("Ingrese el código del aeropuerto: ").strip().upper()

    flights = search_nodes({}, df)
    nodes_df = get_nodes_from_flights(flights)
    src_distances_df = nodes_df.copy()
    src_distances_df.drop(src_distances_df[src_distances_df['Airport Code'] == source].index, inplace=True)
    src_distances_df['Distancia camino minimo'] = None
    src_distances_df.reset_index(inplace=True, drop=True)

    for i in range(len(src_distances_df)):
        distancia_total = flight_graph.dijkstra_distance(source, src_distances_df.at[i, 'Airport Code'])
        if distancia_total:
            src_distances_df.at[i, 'Distancia camino minimo'] = distancia_total

    src_distances_df = src_distances_df.sort_values(by='Distancia camino minimo', ascending=False)
    src_distances_df = src_distances_df.head(10)
    src_distances_df.reset_index(inplace=True, drop=True)

    for i in range(len(src_distances_df)):
        print(f"Top {(i + 1)}")
        print(f"  - Codigo: {src_distances_df.at[i, 'Airport Code']}")
        print(f"  - Nombre: {src_distances_df.at[i, 'Airport Name']}")
        print(f"  - Ciudad: {src_distances_df.at[i, 'Airport City']}")
        print(f"  - Pais: {src_distances_df.at[i, 'Airport Country']}")
        print(f"  - Latitud: {src_distances_df.at[i, 'Airport Latitude']}")
        print(f"  - Longitud: {src_distances_df.at[i, 'Airport Longitude']}")
        numero_redondeado = round(src_distances_df.at[i, 'Distancia camino minimo'], ndigits=2)
        print(f"  - Distancia camino minimo: {numero_redondeado}")

if __name__ == "__main__":
    main()

