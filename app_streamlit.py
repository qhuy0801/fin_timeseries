import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Times Series Signal Processing",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={"About": "qhuy0168@gmail.com"},
)

with st.sidebar:
    selected = option_menu(
        menu_title="Main menu",
        menu_icon="cast",
        options=["QCOM_LSTM"],
        icons=["cpu"],
        default_index=0,
    )

exec(open(f"./application/main/frontend/modules/{selected.lower()}.py").read())
