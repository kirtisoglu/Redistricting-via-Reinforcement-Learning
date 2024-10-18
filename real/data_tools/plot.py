


import pandas as pd
import matplotlib as plt
import plotly.graph_objects as go
import plotly.express as px
import branca.colormap as cm
import branca
from .DataHandler import DataHandler
from gerrychain.partition import Partition
import folium
import folium.plugins
import geopandas



" Does not work. Needs adjustments. "

class Plot:
    " Plots initial and final solutions side by side with zoom option "

    def __init__(self, geo_data, geo_candidates) -> None:

        #self._create_properties()
        #self.show = self.visualize()
        
        self.geo_data = geo_data  # name this as data
        self.geo_candidates = geo_candidates

            

    def basemap(self):

        fig = px.scatter_mapbox(self.geo_data, lat="lat", lon="lon", hover_name="City", hover_data=["State", "Population"],
                                color_discrete_sequence=["fuchsia"], zoom=3, height=300)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        return fig
    
    
    def incomplete():
        return


    def plot(self, data, centers, attribute: str, color=None, fake_center=None):
        # Ensure data[attribute] has appropriate type for indexing
        data[attribute] = pd.Categorical(data[attribute])
        
        # Map each cluster to a color using a cycle of the Plotly qualitative palette
        colors = px.colors.qualitative.Plotly  # This is an example palette
        color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(data[attribute].cat.categories)}
        data['color'] = data[attribute].map(color_map)

        fig = px.choropleth_mapbox(
            data,
            geojson=data.geometry.__geo_interface__,
            locations=data.index,
            color=data['color'],
            mapbox_style="open-street-map",
            center={"lat": data.geometry.centroid.y.mean(), "lon": data.geometry.centroid.x.mean()},
            height=800,
            zoom=10,
            opacity=0.5,
            color_discrete_map="identity",  # Ensure this uses the direct mapping of assigned colors
            hover_data=[data['pop']]  # Show population data on hover
        )

        # Add cluster centers as markers
        for center in centers:
            center_point = data.loc[center].geometry.centroid
            fig.add_scattermapbox(
                lat=[center_point.y],
                lon=[center_point.x],
                mode='markers',
                marker=dict(size=10, color='black'),  # Black markers for centers
                name=f'District={center}'
            )

        return fig.show()
    


    def compare(self, initial_partition: Partition, final_partition: Partition):
        
        "plots initial and final partitions of recomb chain side by side"
        
        centers = self.geo_candidates.loc[self.geo_candidates['block'].isin(final_partition.centers.values())] 
        others = self.geo_candidates.loc[~self.geo_candidates['block'].isin(final_partition.centers.values())]
        
        self.geo_data['initial_district'] = [final_partition.assignment[node] for node in self.geo_data.index]
        self.geo_data['final_district'] = [initial_partition.assignment[node] for node in self.geo_data.index]
        regions_initial = self.geo_data.dissolve(by='initial_district', as_index=False)
        regions_final = self.geo_data.dissolve(by='final_district', as_index=False) 
        
        regions_initial_new = regions_initial.copy()
        regions_final_new = regions_final.copy()
        regions_initial_new['color'] = [x % 10 for x in range(100)]
        regions_final_new['color'] = [x % 10 for x in range(100)]
        del regions_initial_new['centroid']
        del regions_final_new['centroid']
        regions_initial_json = regions_initial_new.to_json()
        regions_final_json = regions_final_new.to_json()

        step = cm.linear.Paired_10.scale(0, 9).to_step(10)

        # empty figures side by side
        fig = branca.element.Figure()
        subplot1 = fig.add_subplot(1, 2, 1)
        subplot2 = fig.add_subplot(1, 2, 2)

        # INITIAL
        m = folium.Map([41.85, -87.68], zoom_start=10, tiles="OpenStreetMap")
        folium.GeoJson(
            regions_initial_json,
            name="Initial Plan",
            tooltip=folium.GeoJsonTooltip(fields=["initial_district"]),
            popup=folium.GeoJsonPopup(fields=["initial_district"]),
            style_function=lambda feature: {
                "fillColor": step(feature['properties']['color']),
                "color": "black",
                "weight": 2,
                "fillOpacity": 0.5,
            },
            highlight_function=lambda x: {"fillOpacity": 0.8},
        ).add_to(m)
        
        self.geo_candidates.explore(
            m=m,  # pass the map object
            color='black',  
            name="candidates",  # name of the layer in the map
        )

        folium.plugins.ScrollZoomToggler().add_to(m) #Adds a button to enable/disable zoom scrolling

        folium.plugins.Fullscreen(   # To make the map full screen
            position="topright",
            title="Expand me",
            title_cancel="Exit me",
            force_separate_button=True,
        ).add_to(m)

        folium.LayerControl().add_to(m)  # use folium to add layer control
        
        # FINAL
        f = folium.Map([41.85, -87.68], zoom_start=10, tiles="OpenStreetMap")
        folium.GeoJson(
            regions_final_json,
            name="Final Plan",
            tooltip=folium.GeoJsonTooltip(fields=["final_district"]),
            popup=folium.GeoJsonPopup(fields=["final_district"]),
            style_function=lambda feature: {
                "fillColor": step(feature['properties']['color']),
                "color": "black",
                "weight": 2,
                "fillOpacity": 0.5,
            },
            highlight_function=lambda x: {"fillOpacity": 0.8},
        ).add_to(f)
        
        # To make the map full screen
        folium.plugins.Fullscreen(
            position="topright",
            title="Expand me",
            title_cancel="Exit me",
            force_separate_button=True,
        ).add_to(f)

        centers.explore(
            m = f,  # pass the map object
            color='red',  
            name="centers",  # name of the layer in the map
        )
        others.explore(
            m = f,  # pass the map object
            color='black',  
            name="others",  # name of the layer in the map
        )

        folium.LayerControl().add_to(f)  # use folium to add layer control
        
        #add them to the empty figures
        subplot1.add_child(m)
        subplot2.add_child(f)

        return fig, regions_initial_new, regions_final_new, centers, others



    

