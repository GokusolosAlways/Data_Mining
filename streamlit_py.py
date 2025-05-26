
import streamlit as st
import pickle
import numpy as np
import xgboost

# ----- 1. Charger le modèle
with open("xgboost_model.pkl", "rb") as f:
    model1 = pickle.load(f)
model = model1[0]
st.title("Prédiction du prix de base d'un bus")

# ----- 2. Mappings pour les variables catégorielles
# Mapping pour ‘Source Type’
source_type_map = {
    "Purchasing Cooperative": 0,
    "State Contract": 1
}

# Mapping pour ‘Bus Manufacturer’
bus_manufacturer_map = {
    "BYD": 0,
    "Blue Bird": 1,
    "Collins Bus": 2,
    "Endera": 3,
    "GreenPower": 4,
    "IC Bus": 5,
    "Lightning eMotors/Collins Bus": 6,
    "Lion Electric": 7,
    "Magellan": 8,
    "Micro Bird": 9,
    "Pegasus/Zeus Electric": 10,
    "Thomas Built Buses": 11,
    "Trans Tech": 12,
    "nan": 13
}

# Mapping pour ‘Bus Model’
bus_model_map = {
    "051MS": 0,
    "All-American": 1,
    "Atlas/Z19": 2,
    "BEAST": 3,
    "CE/PB10E": 4,
    "DE516": 5,
    "DE516F": 6,
    "DH500": 7,
    "G5": 8,
    "LionC": 9,
    "LionD": 10,
    "Magellan": 11,
    "Nano BEAST": 12,
    "O-Series": 13,
    "S11N01": 14,
    "S12N01": 15,
    "S12N02": 16,
    "S82N01": 17,
    "SST": 18,
    "Saf-T-Liner C2 Jouley": 19,
    "Vision": 20,
    "nan": 21
}

# Mapping pour ‘Bus Type’
bus_type_map = {
    "Type A": 0,
    "Type C": 1,
    "Type D": 2
}

# Mapping pour ‘State’
state_map = {
    "AR": 0,  "AZ": 1,  "FL": 2,  "GA": 3,
    "KY": 4,  "LA": 5,  "ME": 6,  "MS": 7,
    "NC": 8,  "NY": 9,  "OH": 10, "SC": 11,
    "UT": 12, "VA": 13, "WA": 14, "WV": 15
}

# Mapping pour ‘Vehicle Dealer’
vehicle_dealer_map = {
    "BYD Coach & Bus LLC": 0,
    "Blue Bird Bus Sales of Pittsburgh": 1,
    "Bluegrass International": 2,
    "Boyd-Cat": 3,
    "Bryson Sales": 4,
    "Bryson Sales and Service": 5,
    "Burroughs Diesel": 6,
    "Canyon State Bus Sales": 7,
    "Carolina International Trucks": 8,
    "Carolina Thomas": 9,
    "Central States Bus Sales": 10,
    "Empire Truck Sales": 11,
    "Florida Transportation Systems": 12,
    "GreenPower Motor Company": 13,
    "GreenPower of WV": 14,
    "Gregory Poole": 15,
    "HK Truck Services": 16,
    "Interstate Transportation Equipment": 17,
    "Kent-Mitchell Bus Sales and Service": 18,
    "Kingmor Supply": 19,
    "Leonard Bus Sales": 20,
    "Leonard Bus Sales / AT New York City": 21,
    "Leonard Bus Sales / Truck King International": 22,
    "Lewis Bus Group": 23,
    "Lion Electric": 24,
    "Master's Transportation": 25,
    "Matheny Motor Truck": 26,
    "Matthers Bus Alliance": 27,
    "Matthews Bus Alliance": 28,
    "Matthews Buses / Nesco Bus and Truck Sales": 29,
    "Midwest Bus Sales": 30,
    "NY Bus Sales / JP Bus & Truck Repair": 31,
    "NY Bus Sales / JP Bus and Truck Repair": 32,
    "Northwest Bus Sales": 33,
    "Ohio CAT": 34,
    "Peach State Truck Centers": 35,
    "RWC Group": 36,
    "RWC International": 37,
    "Rohrer Enterprises": 38,
    "Ross Bus and Equipment Sales": 39,
    "Rush Truck Centers of Georgia": 40,
    "Rush Truck Sales": 41,
    "Schetky NW Sales": 42,
    "Sonny Merryman": 43,
    "Sunstate International Trucks": 44,
    "Twin State Trucks": 45,
    "WNY Bus Parts": 46,
    "WNY Bus Parts / Factory Direct Bus Sales": 47,
    "Waters Truck and Tractor": 48,
    "Whites IC Bus": 49,
    "Worldwide Equipment of WV": 50,
    "Yancey Bros.": 51
}

# Mapping pour ‘Source’
source_map = {
    "Arizona Department of Administration": 0,
    "Arkansas Department of Transformation and Shared Services": 1,
    "Florida Department of Education": 2,
    "Georgia Department of Administrative Services": 3,
    "Kentucky Department of Education": 4,
    "Louisiana Division of Administration": 4,
    "META Co-Op": 6,
    "Maine Department of Administrative and Financial Services": 7,
    "Mississippi Department of Education": 8,
    "New York State Office of General Services": 9,
    "North Carolina Department of Administration": 10,
    "South Carolina Division of Procurement Services": 11,
    "Utah Division of Purchasing and General Services": 12,
    "Virginia Department of General Services": 13,
    "Washington Office of Superintendent of Public Instruction": 14,
    "West Virginia Purchasing Division": 15
}

# Mapping pour ‘Maintenance_History’
maintenance_history_map = {
    "Average": 0,
    "Good": 1,
    "Poor": 2
}

# Mapping pour ‘Transmission_Type’
transmission_type_map = {
    "Automatic": 0,
    "Manual": 1
}

# Mapping pour ‘Owner_Type’
owner_type_map = {
    "First": 0,
    "Third": 1
}

# Mapping pour ‘Tire_Condition’
tire_condition_map = {
    "Good": 0,
    "New": 1,
    "Worn Out": 2
}

# Mapping pour ‘Brake_Condition’
brake_condition_map = {
    "Good": 0,
    "New": 1,
    "Worn Out": 2
}

# Mapping pour ‘Battery_Status’
battery_status_map = {
    "Good": 0,
    "New": 1,
    "Weak": 2
}

# ----- 3. Widgets Streamlit
st.subheader("Variables catégorielles")
st.selectbox("Source Type", list(source_type_map.keys()), key="Source Type")
st.selectbox("Bus Manufacturer", list(bus_manufacturer_map.keys()), key="Bus Manufacturer")
st.selectbox("Bus Model", list(bus_model_map.keys()), key="Bus Model")
st.selectbox("Bus Type", list(bus_type_map.keys()), key="Bus Type")
st.selectbox("State", list(state_map.keys()), key="State")
st.selectbox("Vehicle Dealer", list(vehicle_dealer_map.keys()), key="Vehicle Dealer")
st.selectbox("Source", list(source_map.keys()), key="Source")
st.selectbox("Maintenance History", list(maintenance_history_map.keys()), key="Maintenance_History")
st.selectbox("Transmission Type", list(transmission_type_map.keys()), key="Transmission_Type")
st.selectbox("Owner Type", list(owner_type_map.keys()), key="Owner_Type")
st.selectbox("Tire Condition", list(tire_condition_map.keys()), key="Tire_Condition")
st.selectbox("Brake Condition", list(brake_condition_map.keys()), key="Brake_Condition")
st.selectbox("Battery Status", list(battery_status_map.keys()), key="Battery_Status")

st.subheader("Variables numériques")
purchase_year       = st.number_input("Purchase Year", min_value=2000, max_value=2030, value=2020)
seating_capacity    = st.number_input("Seating Capacity", min_value=1, max_value=100, value=30)
vehicle_age         = st.number_input("Vehicle Age (years)", min_value=0, value=1)
engine_size         = st.number_input("Engine Size (L)", min_value=0.0, step=0.1, value=4.0)
insurance_premium   = st.number_input("Insurance Premium ($)", min_value=0.0, value=1000.0)
fuel_efficiency     = st.number_input("Fuel Efficiency (mpg)", min_value=0.0, value=10.0)

# ----- 4. Encodage des sélections
X = [
    source_type_map[st.session_state["Source Type"]],
    purchase_year,
    bus_manufacturer_map[st.session_state["Bus Manufacturer"]],
    bus_model_map[st.session_state["Bus Model"]],
    bus_type_map[st.session_state["Bus Type"]],
    seating_capacity,
    state_map[st.session_state["State"]],
    vehicle_dealer_map[st.session_state["Vehicle Dealer"]],
    source_map[st.session_state["Source"]],
    vehicle_age,
    engine_size,
    insurance_premium,
    fuel_efficiency,
    maintenance_history_map[st.session_state["Maintenance_History"]],
    transmission_type_map[st.session_state["Transmission_Type"]],
    owner_type_map[st.session_state["Owner_Type"]],
    tire_condition_map[st.session_state["Tire_Condition"]],
    brake_condition_map[st.session_state["Brake_Condition"]],
    battery_status_map[st.session_state["Battery_Status"]],
]

# ----- 5. Prédiction
if st.button("Prédire le Base Price"):
    arr = np.array([X])
    pred = model.predict(arr)[0]
    st.success(f"Prix de base estimé : {pred:,.2f} $")

