import osmnx as ox
import taxicab as tc
import matplotlib.pyplot as plt
from streamlit_geolocation import streamlit_geolocation
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title='Маршрут',
    page_icon="🗺️"
)
st.write('Построить маршрут от меня')
location = streamlit_geolocation()
coord_start = [location['latitude'], location['longitude']]

d = pd.read_csv('./static/geo.csv')

city = st.selectbox(
    'Выберете город:',
    d['City'].unique())
mem = st.selectbox(
    'Выберете достопримечательность:',
    d[(d['City'] == city) & (~d['Lon'].isna()) & (~d['Lat'].isna())]['Name'])

coord_end = d[(d['City'] == city) & (d['Name'] == mem)][['Lat', 'Lon']].values.tolist()[0]

if coord_start[0] is not None and coord_start[1] is not None:
    G = ox.graph_from_point(coord_start, dist=3000, network_type='walk')
    G = ox.speed.add_edge_speeds(G)
    G = ox.speed.add_edge_travel_times(G)

    orig = coord_start
    dest = coord_end
    route = tc.distance.shortest_path(G, orig, dest)

    fig, ax = tc.plot.plot_graph_route(G, route, node_size=15, show=False, close=False, figsize=(10, 10))
    padding = 0.005
    ax.scatter(orig[1], orig[0], c='lime', s=200, label='orig', marker='x')
    ax.scatter(dest[1], dest[0], c='red', s=200, label='dest', marker='x')
    ax.set_ylim([min([orig[0], dest[0]])-padding, max([orig[0], dest[0]])+padding])
    ax.set_xlim([min([orig[1], dest[1]])-padding, max([orig[1], dest[1]])+padding])
    st.pyplot(fig)
