import streamlit as st
import utils as fct
import pandas as pd

st.title("Euroleague Playoffs Prediction")

T_col1, T_col2 = st.columns([2, 1])

with T_col1:
    st.write("Author : Titouan Houde - Sport Data Scientist")

with T_col2:
    st.markdown("About me : [LinkedIn](https://www.linkedin.com/in/titouan-houde/)")

options = ["Olympiacos Piraeus", "FC Barcelona", "Real Madrid", "AS Monaco", 
           "Maccabi Playtika Tel Aviv", "Partizan Mozzart Bet Belgrade", "Zalgiris Kaunas", "Fenerbahce Beko Istanbul"]

logos = {
    "Olympiacos Piraeus" : "./assets/olympiacos.png",
    "FC Barcelona": "./assets/barcelona.png",
    "Real Madrid": "./assets/real_madrid.png",
    "AS Monaco" : "./assets/monaco.png",
    "Maccabi Playtika Tel Aviv" : "./assets/maccabi.png",
    "Partizan Mozzart Bet Belgrade" : "./assets/partizan.png",
    "Zalgiris Kaunas" : "./assets/zalgiris.png",
    "Fenerbahce Beko Istanbul" : "./assets/fener.png",
    "vs" : "./assets/vs.jpg",
    "euroleague" : "./assets/euroleague.png"
}


# Coeff to predict : 
Team = ["Olympiacos Piraeus", "FC Barcelona", "Real Madrid", "AS Monaco",
         "Maccabi Playtika Tel Aviv", "Partizan Mozzart Bet Belgrade", "Zalgiris Kaunas", "Fenerbahce Beko Istanbul"]

st.write("")
selected_option1 = st.selectbox("Select a Home Team :", options)

# Remove selected option from options list for the second selectbox
options2 = [option for option in options if option != selected_option1]
selected_option2 = st.selectbox("Select an Away Team :", options2)

st.markdown("---")

logo_filename1 = logos[selected_option1]
vs = logos["vs"]
logo_filename2 = logos[selected_option2]
logo_euroleague = logos["euroleague"]

# Predict columns
col1, col2, col3 = st.columns(3)

with col1:
    st.image(logo_filename1, use_column_width=True)

with col2:
    st.image(vs, use_column_width=True)
# Display the second logo in the second column
with col3:
    st.image(logo_filename2, use_column_width=True)

st.markdown("---")

sl_col1, sl_col2, sl_col3,  = st.columns(3)

with sl_col1:
    
    st.write("Performance in offense ?")
    HomeOffense = st.select_slider(label = "", label_visibility= 'collapsed',
                       options = ["Worst", "Badly", "Not very well", "Average", "Well", "Very well", "Best"],
                       value = "Average")
    
    st.write("In defense ?")
    HomeDefense = st.select_slider(label = " ", label_visibility= 'collapsed',
                       options = ["Worst", "Badly", "Not very well", "Average", "Well", "Very well", "Best"],
                       value = "Average")


with sl_col3:
    
    st.write("Performance in offense ?")
    AwayOffense = st.select_slider(label = "  ", label_visibility= 'collapsed',
                       options = ["Worst", "Badly", "Not very well", "Average", "Well", "Very well", "Best"],
                       value = "Average")
    
    st.write("In defense ?")
    AwayDefense = st.select_slider(label = "   ", label_visibility= 'collapsed',
                       options = ["Worst", "Badly", "Not very well", "Average", "Well", "Very well", "Best"],
                       value = "Average")

data = pd.read_parquet('./assets/data.parquet.gzip')
data1 = pd.read_parquet('./assets/data1.parquet.gzip')
data2 = pd.read_parquet('./assets/data2.parquet.gzip')
from sklearn.svm import SVC
model = SVC(kernel= 'rbf', probability= True)
model.fit(data1, data2)

if st.button('Predict !', use_container_width=True):

    coefficient = []

    for V in [HomeOffense, HomeDefense] :

        if V == "Worst":
            coefficient.append(-0.6)
        elif V == "Badly":
            coefficient.append(-0.4)
        elif V == "Not very well":
            coefficient.append(-0.2)
        elif V == "Average":
            coefficient.append(0)
        elif V == "Well":
            coefficient.append(0.125)
        elif V == "Very well":
            coefficient.append(0.35)
        else:  # "Best"
            coefficient.append(0.6)

    for V in [AwayOffense, AwayDefense] :

        if V == "Worst":
            coefficient.append(-0.5)
        elif V == "Badly":
            coefficient.append(-0.3)
        elif V == "Not very well":
            coefficient.append(-0.15)
        elif V == "Average":
            coefficient.append(0)
        elif V == "Well":
            coefficient.append(0.2)
        elif V == "Very well":
            coefficient.append(0.4)
        else:  # "Best"
            coefficient.append(0.7)

    proba0, proba1, pred = fct.predict_matchup(selected_option1, selected_option2, 
                                               coefficient[0], coefficient[0],
                                                coefficient[1], coefficient[2],
                                                coefficient[2], coefficient[3], data, model)

    if pred == 1 :
        vainqueur = selected_option1
        with sl_col2 :
            st.image(logo_filename1, use_column_width= True)
    else : 
        vainqueur = selected_option2
        with sl_col2 :
            st.image(logo_filename2, use_column_width= True)

    vainqueur_display = st.markdown("**According to the parameters, "+str(vainqueur)+ " has more chance to win the game.**")

st.write("NB: If you do several trials, you will see that the attack makes the prediction vary much more. This is because the statistics describe the offense more than the defense, the results would be different with more specific data.")
st.markdown("---")

st.subheader("Why this app ?")

st.write("In a high performance context, data can be used to obtain details that will sometimes make the difference.")

st.write("This app is a fun way to use data but it is possible to do much more, like describing the style of play of different teams and also to get the factors that lead teams to the victory, which is essential for a staff.")

st.write("This app will be completed with other analysis that will allow to describe teams and players.")

