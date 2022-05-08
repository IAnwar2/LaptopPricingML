import streamlit as st
import pickle
import numpy as np

def loadModel():
    with open ('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = loadModel()

regressor = data["model"]
le_processor_name = data["le_processor_name"]
le_ram_gb = data["le_ram_gb"]
le_weight = data["le_weight"]
le_ssd = data["le_ssd"]
le_os = data["le_os"]
le_os_bit = data["le_os_bit"]
le_display_size = data["le_display_size"]
le_Touchscreen = data["le_Touchscreen"]

def ShowPage():
    st.title("Average prices for Laptops")

    st.write("""### Enter Laptop specs to check if you're getting ripped off""")

    processorNames = {

        "Core i5",    
        "Core i3",  
        "Core i7",   
        "Ryzen 5",     
        "Ryzen 7",   
        "M1",          
        "Ryzen 9",     
        "Ryzen 3",         
        "other"     
    }

    RamNum = {
        "8 GB GB",
        "16 GB GB",
        "4 GB GB",
        "other"
    }

    weight = {
        "Casual",
        "ThinNlight",
        "Gaming"
    }

    ssd = {
        "512 GB",
        "256 GB",
        "1024 GB",
        "0 GB",
        "other"
    }

    os = {
        "Windows",
        "Mac",
        "DOS"
    }

    osBit = {
        "64-bit",
        "32-bit"
    }

    graphicGB = {
        "0",
        "2",
        "4",
        "6",
        "8"
    }

    displaySize = {
        "15.6",
        "16",
        "14",
        "13.3",
        "other"
    }

    touch = {
        "No",
        "Yes"
    }

    processorNames = st.selectbox("Processor Name", processorNames)
    RamNum = st.selectbox("How many GB RAM", RamNum)
    weight = st.selectbox("Weight", weight)
    ssd = st.selectbox("SSD", ssd)
    os = st.selectbox("OS", os)
    osBit = st.selectbox("OS Bit", osBit)
    graphicGB = st.selectbox("How many GB in the graphics card (select 0 if no graphics card)", graphicGB)
    displaySize = st.selectbox("Display Size", displaySize)
    touch = st.selectbox("TouchScreen", touch)

    send = st.button("Calculate Price")

    if send:
        X = np.array([[processorNames, RamNum, ssd, os, osBit, graphicGB, weight, displaySize, touch]])
        X[:, 0] = le_processor_name.transform(X[:,0])
        X[:, 1] = le_ram_gb.transform(X[:,1])
        X[:, 2] = le_ssd.transform(X[:,2])
        X[:, 3] = le_os.transform(X[:,3])
        X[:, 4] = le_os_bit.transform(X[:,4])
        X[:, 6] = le_weight.transform(X[:,6])
        X[:, 7] = le_display_size.transform(X[:,7])
        X[:, 8] = le_Touchscreen.transform(X[:,8])
        X = X.astype(float)

        price = regressor.predict(X)

        price[0] *= 0.017
        st.subheader(f"The estimated retail cost(CAD) is ${price[0]:0.2f}")

