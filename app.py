import streamlit as st

import pickle as pk

with open("model_linear.bin", "rb") as f_in:
    dv, model = pk.load(f_in)
    
def prediction(details):
    if not isinstance(details, dict):
        print("invalid")
        return
    print(details)
    X_val = dv.transform(details)
    print(X_val)
    try:
        price = model.predict(X_val) 
        return price
    except Exception as e:
        return "Error processing this request please try again"
    
    
    # X_val = dv.

st.title("House Sale prediction model")
col1, col2 = st.columns(2)
with col1:

# 1. MSSubClass
    ms_subclass = st.number_input("MSSubClass:the type of dwelling", min_value=1, max_value=200, step=1)

    # 2. MSZoning
    ms_zoning = st.selectbox("MSZoning:general zoning classification", ["RL", "RM", "C (all)", "FV", "RH"])

    # 3. LotArea
    lot_area = st.number_input("LotArea (sq ft): Lot size", min_value=0.0, step=1.0)

    # 4. LotConfig
    lot_config = st.selectbox("LotConfig", ["Inside", "Corner", "CulDSac", "FR2", "FR3"])

    # 5. BldgType
    bldg_type = st.selectbox("BldgType: Type of dwelling", ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"])
with col2:
    # 6. OverallCond
    overall_cond = st.slider("OverallCond:  condition of the house", min_value=1, max_value=10, step=1)

    # 7. YearBuilt
    year_built = st.number_input("YearBuilt", min_value=1800, max_value=2024, step=1)

    # 8. YearRemodAdd
    year_remod_add = st.number_input("YearRemodAdd: Remodel date", min_value=1800, max_value=2024, step=1)

    # 9. Exterior1st[]
#     'VinylSd-' 'MetalSd-' 'Wd Sdng-' 'HdBoard-' 'BrkFace-' 'WdShing-' 'CemntBd-'
#  'Plywood-' 'AsbShng-' 'Stucco-' 'BrkComm-' 'AsphShn-' 'Stone' 'ImStucc'
#  'CBlock
    exterior_1st = st.selectbox("Exterior1st: exterior covering", ["AsbShng", "BrkComm", "CemntBd", "HdBoard", "MetalSd", "Plywood", "Stucco", "VinylSd", "Wd Sdng", "WdShing", "BrkFace", "AsphShn", "Stone", "ImStucc", "CBlock"])

    # 10. BsmtFinSF2
    bsmt_fin_sf2 = st.number_input("BsmtFinSF2 (sq ft): finished square feet", min_value=0.0, step=1.0)

    # 11. TotalBsmtSF
    total_bsmt_sf = st.number_input("TotalBsmtSF (sq ft): Total square feet of basement area", min_value=0.0, step=1.0)


if st.button("Submit"):
    st.write("You click me 😂😂")
    # validation
    if not ms_subclass or not lot_area or not overall_cond or not year_built or not year_remod_add or not bsmt_fin_sf2:
        st.warning("All fields must be filled out before submitting.")
    else:
       
        input_data = { "MSSubClass": ms_subclass, "MSZoning": ms_zoning, "LotArea": lot_area, "LotConfig": lot_config, "BldgType": bldg_type, "OverallCond": overall_cond, "YearBuilt": year_built, "YearRemodAdd": year_remod_add, "Exterior1st": exterior_1st, "BsmtFinSF2": bsmt_fin_sf2, "TotalBsmtSF": total_bsmt_sf, }
        result = prediction(details = input_data)

        st.write(f'The price of this property is extimated to be {result}')
    





# Displaying the input values
# st.write("### Input Summary")
# st.write(f"**MSSubClass:** {ms_subclass}")
# st.write(f"**MSZoning:** {ms_zoning}"[])
# st.write(f"**LotArea:** {lot_area}")
# st.write(f"**LotConfig:** {lot_config}")
# st.write(f"**BldgType:** {bldg_type}")
# st.write(f"**OverallCond:** {overall_cond}")
# st.write(f"**YearBuilt:** {year_built}")
# st.write(f"**YearRemodAdd:** {year_remod_add}")
# st.write(f"**Exterior1st:** {exterior_1st}")
# st.write(f"**BsmtFinSF2:** {bsmt_fin_sf2}")
