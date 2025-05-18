import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import networkx as nx

data = pd.read_csv('data/cleanedData.csv', parse_dates=['arrival_date'])

st.sidebar.header('filter guests')

min_guests = int(data['total_guests'].min())
max_guests = int(data['total_guests'].max())
guest_range = st.sidebar.slider('Total Guests', min_guests, max_guests, (min_guests, max_guests))

#booking method
channel = st.sidebar.multiselect('Distribution Channel', data['distribution_channel'].unique())

#purpose of visit
stay_purpose_map = {
    'Business' : ['Corporate'],
    'Leisure' : ['Online TA', 'Offline TA/TO']
}
stay_purpose = st.sidebar.multiselect('Stay Purpose', list(stay_purpose_map.keys()))

#season of visit
season = st.sidebar.multiselect('season', data['season'].unique())

#room type
room_type = st.sidebar.multiselect('Assigned room type', data['assigned_room_type'].unique())

#length of visit
stay_min = int(data['stay_length'].min())
stay_max = int(data['stay_length'].max())
stay_range = st.sidebar.slider('Stay length', stay_min, stay_max, (stay_min, stay_max))

#loyal that is repeated guests
loyal = st.sidebar.radio('Loyalty Program', ['Both', 'Yes', 'No'])

country = st.sidebar.multiselect('country', data['country'].unique())

filtered_data = data.copy()

filtered_data = filtered_data[(filtered_data['total_guests'] >= guest_range[0]) &
                               (filtered_data['total_guests'] <= guest_range[1])]


if channel:
    filtered_data = filtered_data[filtered_data['distribution_channel'].isin(channel)]

if stay_purpose:
    purpose_segments = sum([stay_purpose_map[k] for k in stay_purpose], [])
    filtered_data = filtered_data[filtered_data['market_segment'].isin(purpose_segments)]

if season:
    filtered_data = filtered_data[filtered_data['season'].isin(season)]

if room_type:
    filtered_data = filtered_data[filtered_data['assigned_room_type'].isin(room_type)]
filtered_data = filtered_data[(filtered_data['stay_length'] >= stay_range[0]) &
                               (filtered_data['stay_length'] <= stay_range[1])]

if loyal != "Both":
    filtered_data = filtered_data[filtered_data['is_repeated_guest'] == (1 if loyal == "Yes" else 0)]

if country:
    filtered_data = filtered_data[filtered_data['country'].isin(country)]

st.title('Hotel Analytics Platform')
st.markdown('---')

#first the metrics
st.subheader('Key Metrics')
st.metric("Average Daily Rate", f"{filtered_data['adr'].mean():.4f}")
st.metric("Average Revenue per Guest", f"{filtered_data['total_revenue'].mean():.4f}")
st.metric("Average Stay Length", f"{filtered_data['stay_length'].mean():.4f} nights")
st.metric("Revenue Efficiency Index", f"{filtered_data['revenue_efficiency_index'].mean():.4f}")
st.metric("Guest Satisfaction Yield", f"{filtered_data['guest_satisfaction_yield'].mean():.4f}")
st.metric("Loyalty Generation Score", f"{filtered_data['loyalty_generation_score'].mean():.4f}")
st.metric("Amnity Utilization Ratio", f"{filtered_data['amenity_utilization_ratio'].mean():.4f}")
st.metric("Operation Excellence metric", f"{filtered_data['operational_excellence_metric'].mean():.4f}")
st.markdown(f"### Number of Guests Matching Current Filters: **{len(filtered_data)}**")

st.markdown('---')

#revenue
st.subheader("Revenue Potential by Guest Segment")
segment_rev = filtered_data.groupby("segment_label")['total_revenue'].sum().sort_values(ascending=False)
fig1, ax1 = plt.subplots()
segment_rev.plot(kind='bar', ax=ax1, color='darkorange')
ax1.set_ylabel("Revenue: ")
st.pyplot(fig1)

# Amenity Usage
st.subheader("Amenity Utilization Ratio")
monthly = filtered_data.copy()
monthly['month'] = monthly['arrival_date'].dt.to_period('M').dt.to_timestamp()
monthly_usage = monthly.groupby('month')['amenity_utilization_ratio'].mean()
fig2, ax2 = plt.subplots()
monthly_usage.plot(ax=ax2, marker='o', color='teal')
ax2.set_ylabel("Amenity Usage Ratio")
st.pyplot(fig2)

# Segment 
st.subheader("Guest Segment Distribution")
segment_counts = filtered_data['segment_label'].value_counts()
fig3, ax3 = plt.subplots()
ax3.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
ax3.axis('equal')
st.pyplot(fig3)

st.subheader("Segment Guests Profile")
segment_features = [
    'lead_time', 'adr', 'stay_length', 'total_revenue',
    'amenity_utilization_ratio', 'promo_sensitivity',
    'simulated_complaints', 'is_repeated_guest'
]

# Radar Plot
segment_means = filtered_data.groupby('guest_segment')[segment_features].mean()
normalized = (segment_means - segment_means.min()) / (segment_means.max() - segment_means.min())
labels = segment_features

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig_radar, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for idx, row in normalized.iterrows():
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, label=f"Segment {idx}")
    ax.fill(angles, values, alpha=0.1)

ax.set_title("Guest Profile based on Segment", y=1.1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_yticklabels([])
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
st.pyplot(fig_radar)

# Forecasted occupancy per room typ
st.subheader("Occupancy Forecast")
forecast_data = pd.read_csv("data/forecast_occupancy.csv", parse_dates=['month'])
filtered_data['month'] = filtered_data['arrival_date'].dt.to_period('M').dt.to_timestamp()
room_history = filtered_data.groupby(['month', 'assigned_room_type']).size().reset_index(name='bookings')
for room in forecast_data['room_type'].unique():
    room_data = forecast_data[forecast_data['room_type'] == room]
    history_data = room_history[room_history['assigned_room_type'] == room]
    fig, ax = plt.subplots()
    ax.plot(history_data['month'], history_data['bookings'], label='Previous', marker='o')
    ax.plot(room_data['month'], room_data['predicted_bookings'], '--', label='Predicted', marker='x')
    ax.set_title(f"Forecast for Room Type {room}")
    ax.set_ylabel("Bookings")
    ax.set_xlabel("Month")
    ax.legend()
    st.pyplot(fig)

# Amenity predictions
st.subheader("Amenity Usage Forecast")
amenity_forecast = pd.read_csv("data/forecast_amenities.csv", parse_dates=['month'])
history_usage = filtered_data.groupby('month')['amenity_utilization_ratio'].mean().reset_index()
fig4, ax4 = plt.subplots()
ax4.plot(history_usage['month'], history_usage['amenity_utilization_ratio'], label='Previous', marker='o')
ax4.plot(amenity_forecast['month'], amenity_forecast['predicted_amenity_usage'], '--', label='Predicted', marker='x', color='purple')
ax4.set_title("Predicted Amenity Usage Ratio")
ax4.set_ylabel("AUR")
ax4.set_xlabel("Month")
ax4.legend()
st.pyplot(fig4)

st.subheader("Guest Journey Flow")

n = 119390 # number of guests to track
#almost the same code as in process.ipynb, just changed to fit into streamlit
journey_data = pd.DataFrame({
    'is_canceled': np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
    'amenity_utilization_ratio': np.random.exponential(5, size=n),
    'simulated_complaints': np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
    'reservation_status': np.random.choice(['Check-Out', 'Canceled'], size=n, p=[0.8, 0.2]),
    'is_repeated_guest': np.random.choice([0, 1], size=n, p=[0.85, 0.15])
})

def build_journey_stages(row):
    stages = ['Booking Confirmed']
    if row['is_canceled'] == 0:
        stages.append('Arrived')
        if row['amenity_utilization_ratio'] > 3:
            stages.append('Used Amenities')
        if row['simulated_complaints'] == 1:
            stages.append('Complained')
        if row['reservation_status'] == 'Check-Out':
            stages.append('Check-Out')
        if row['is_repeated_guest'] == 1:
            stages.append('Repeat Guest')
    else:
        stages.append('Canceled')
    return stages

journey = journey_data.apply(build_journey_stages, axis=1)

edges = Counter()
for path in journey:
    for i in range(len(path) - 1):
        edges[(path[i], path[i + 1])] += 1


G = nx.DiGraph()
for (src, tgt), count in edges.items():
    G.add_edge(src, tgt, weight=count)

node_order = ['Booking Confirmed', 'Arrived', 'Used Amenities', 'Complained', 'Check-Out', 'Repeat Guest', 'Canceled']
pos = nx.shell_layout(G, nlist=[node_order])


fig_journey, ax_journey = plt.subplots(figsize=(12, 10))
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue', edgecolors='black', ax=ax_journey)
nx.draw_networkx_edges(G, pos, arrowstyle='fancy', arrowsize=20, width=2, edge_color='gray', ax=ax_journey)
for node, (x, y) in pos.items():
    ax_journey.text(x, y + 0.05, node, fontsize=10, fontweight='bold', ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))


edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, ax=ax_journey)

ax_journey.set_title("Guest Journey Flow", fontsize=14)
ax_journey.axis('off')
st.pyplot(fig_journey)

#heatmap
st.subheader('Heatmap to visualize realtions between different features')
completed = data[data['is_canceled'] == 0].copy()


completed['satisfaction_score'] = (
    70 + (completed['amenity_utilization_ratio'] * 0.5)
    - (completed['simulated_complaints'] * 10)
)
fig5, ax5 = plt.subplots(figsize=(8, 5))

sns.heatmap(completed[['satisfaction_score', 'adr', 'stay_length', 'total_guests',
                       'simulated_complaints', 'amenity_utilization_ratio']].corr(),
            annot=True, cmap='coolwarm')
ax5.set_title("Satisfaction Relations")
st.pyplot(fig5)

#loyalty in people who complained
who_complained = data[data['simulated_complaints'] > 0]

recovery_rate = who_complained['is_repeated_guest'].mean()
st.subheader(f"Service Recovery Rate: {recovery_rate:.2%} of guests who complained became repeat guests")
fig6, ax6 = plt.subplots(figsize=(8, 5))
sns.barplot(data=who_complained, x='is_repeated_guest', y='amenity_utilization_ratio')
ax6.set_title("Amenity usage among guests who complained (Repeat vs Not)")
ax6.set_xlabel("Repeat Guest")
ax6.set_ylabel("Amenity Utilization Ratio")
st.pyplot(fig6)

upsell_guests = data[
    (data['amenity_utilization_ratio'] > 5) & 
    (data['stay_length'] > 3) & 
    (data['total_of_special_requests'] > 0)
]
st.subheader(f"Guests suitable for upsell offers: {len(upsell_guests)}")