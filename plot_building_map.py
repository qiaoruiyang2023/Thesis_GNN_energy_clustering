import pandas as pd
import os
import folium

# 1. 读取建筑物数据
DATA_PATH = os.path.join('data', 'raw', 'D:/GEO2020_Thesis/energy_cluster_gnn_v02/energy_cluster_gnn/data/raw/buildings_demo.csv')
buildings_df = pd.read_csv(DATA_PATH)

# 2. 创建输出目录
OUTPUT_DIR = os.path.join('results', 'visualization')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. 创建地图，中心点为所有建筑物的平均位置
center_lat = buildings_df['lat'].mean()
center_lon = buildings_df['lon'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='CartoDB positron')

# 4. 为不同类型的建筑物分配颜色
building_types = buildings_df['building_function'].unique()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
color_map = dict(zip(building_types, colors[:len(building_types)]))

# 5. 添加建筑物标记（不聚合，直接加到地图上）
for idx, row in buildings_df.iterrows():
    popup_text = f"""
    <div style='font-family: Arial; font-size: 12px;'>
        <b>Building ID:</b> {row['building_id']}<br>
        <b>Type:</b> {row['building_function']}<br>
        <b>Peak Load:</b> {row['peak_load_kW']:.1f} kW<br>
        <b>Area:</b> {row['area']:.1f} m²<br>
        <b>Height:</b> {row['height']:.1f} m<br>
        <b>Age Range:</b> {row['age_range']}<br>
        <b>Solar:</b> {'Yes' if row['has_solar'] else 'No'}<br>
        <b>Battery:</b> {'Yes' if row['has_battery'] else 'No'}
    </div>
    """
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=5,
        popup=folium.Popup(popup_text, max_width=300),
        color=color_map[row['building_function']],
        fill=True,
        fill_color=color_map[row['building_function']],
        fill_opacity=0.7,
        weight=1
    ).add_to(m)

# 6. 添加图例
legend_html = """
<div style="position: fixed; 
            bottom: 50px; right: 50px; 
            border:2px solid grey; z-index:9999; 
            background-color:white;
            padding: 10px;
            font-family: Arial;
            font-size: 20px;">
<p><b>Building Types</b></p>
"""
for btype, color in color_map.items():
    legend_html += f"<p><span style='color:{color};'>●</span> {btype}</p>"
legend_html += "</div>"
m.get_root().html.add_child(folium.Element(legend_html))

# 7. 添加标题
m.get_root().html.add_child(folium.Element(
    """
    <h3 align='center' style='font-size:20px; font-family:Arial;'><b>Geographic Distribution of Buildings in the Study Area</b></h3>
    """
))

# 8. 保存地图
output_path = os.path.join(OUTPUT_DIR, 'building_locations.html')
m.save(output_path)
print(f"Map saved to: {output_path}")
