import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt
import random
from datetime import datetime, date

# Ustawienia strony
st.set_page_config(layout="wide")
marketing_dict = {'_x1': '_TV', '_x2': '_Facebook', '_x3': '_Onet', '_x4': '_Wp','_x5':'_GW'}
Sp_x = {'Sp_x1':'3500000','Sp_x2':'2500000','Sp_x3':'1500000','Sp_x4':'1000000','Sp_x5':'1500000'}
DMA_dict = {'DMA1':'Warszawa','DMA2':'Kraków','DMA3':'Poznań','DMA4':'Gdańsk', 'DMA5':'Katowice'}
DMA_reve_rate = [0.2, 0.15, 0.3, 0.1, 0.25]

# Generowanie Incentive Revenue Rate krzywą Gaussa z szumem
def generuj_inc_rev_rate(periods, amplitude, frequency, noise_level):
    czas = np.arange(periods)
    smooth_noise = np.cumsum(np.random.normal(0, noise_level, periods))  # Zastosowanie skumulowanego szumu Gaussa
    inc_rev_rate = amplitude * (np.sin(czas / 10 * frequency) + np.cos(czas / 5 * frequency)) + smooth_noise
    
    inc_rev_rate = (inc_rev_rate - np.min(inc_rev_rate)) / (np.max(inc_rev_rate) - np.min(inc_rev_rate))  # Skaluje do zakresu [0, 1]
    inc_rev_rate = inc_rev_rate / np.sum(inc_rev_rate)  # Normalizuje, aby suma wynosiła 1
    return inc_rev_rate

# Generowanie w oparciu o krzywą Gaussa
def generuj_sezonowosc(periods, amplitude, mean, std_dev):
    global sezonowosc
    czas = np.arange(periods)
    sezonowosc = amplitude * np.exp(-0.5 * ((czas - mean) / std_dev) ** 2)
    sezonowosc = (sezonowosc - np.min(sezonowosc)) / (np.max(sezonowosc) - np.min(sezonowosc))  # Skaluje do zakresu [0, 1]
    sezonowosc = sezonowosc / np.sum(sezonowosc)  # Normalizuje, aby suma wynosiła 1
    return sezonowosc

def random_gen(periods):
    _min = 0.01
    _max = 0.95
    numbers_ = periods
    
    # Generowanie losowych liczb z przedziału 0.05 do 0.95
    random_numbers = np.random.uniform(_min, _max, numbers_)
    return random_numbers

def random_gen_rev(periods):
    _min = 0.01
    _max = 0.95
    numbers_ = periods
    
    # Generowanie losowych liczb z przedziału 0.05 do 0.95
    random_numbers_rev = np.random.uniform(_min, _max, numbers_)
    random_numbers_rev = (random_numbers_rev / random_numbers_rev.sum()) * 1
    return random_numbers_rev

def random_select(reve_rate_list):
    selected_value = random.choice(reve_rate_list)
    reve_rate_list.remove(selected_value)
    return selected_value

def Data_T3(base_sales_total, periods, Sp_x, marketing_dict, DMA_dict):
    global df_T31
       
    # Tworzenie DataFrame z kolumnami -> przebudowane działa poprawnie
    df_T3 = pd.DataFrame({
            'Time Period': [f'P{i+1}' for i in range(periods)],
            'Base_S_Plan_rate': sezonowosc,
            'Base_S': sezonowosc * base_sales_total,
            'Inc_rev_rate': generuj_inc_rev_rate(periods, amplitude, frequency, noise_level),
    })
    # Obliczanie wartości w kolumnie 'Inc_reve' jako iloczyn 'Base_S' i 'Inc_rev_rate' 
    df_T3['Inc_reve'] = df_T3['Base_S'] * df_T3['Inc_rev_rate']

    # Obliczanie sumy wartości w słowniku Sp_x -> wyniki ok
    sum_sp_x_dict = sum(float(value) for value in Sp_x.values())
    st.write(f"Suma wartości w słowniku Sp_x: {sum_sp_x_dict:,.2f}")
    #print(f"Suma wartości w słowniku Sp_x: {sum_sp_x_dict:,.2f}")
    # Obliczanie sumy kolumny Inc_reve > wyniki ok
    sum_inc_reve = df_T3['Inc_reve'].sum()
    st.write(f"Suma kolumny Inc_reve: {sum_inc_reve:,.2f}")
    #print(f"Suma kolumny Inc_reve: {sum_inc_reve:,.2f}")

    # Dodawanie kolumn `DMA1_BS` zgodnie ze słownikiem DMA_dict -> to zostało przebudowane i działą poprawnie
    reve_rate_copy = DMA_reve_rate.copy()
    for dma_key in DMA_dict.keys():
        if len(reve_rate_copy) == 0:
            reve_rate_copy = DMA_reve_rate.copy()  # Reset listy, jeśli wszystkie wartości zostały wykorzystane
        DMA_distribution_factor = random_select(reve_rate_copy)
        df_T3[f'{dma_key}_BS'] = DMA_distribution_factor * df_T3['Base_S']

    # Dodawanie brakujących kolumn `Sp_r_x` -> to działa dobrze
    for key in Sp_x.keys():
        column_name = f'Sp_{key[-2:]}'
        df_T3[f'Sp_r_{key[-2:]}'] = random_gen_rev(periods)

    # Dystrybucja wartości w kolumnach Inc_reve na poszczególne DMA kolumny -> przebudowane działa poprawnie
    reve_rate_copy = DMA_reve_rate.copy()
    for dma_key in DMA_dict.keys():
        if len(reve_rate_copy) == 0:
            reve_rate_copy = DMA_reve_rate.copy()  # Reset listy, jeśli wszystkie wartości zostały wykorzystane
        DMA_distribution_factor = random_select(reve_rate_copy)
        df_T3[f'{dma_key}_Inc_rev'] = DMA_distribution_factor * df_T3['Inc_reve']

    # Dodawanie i aktualizacja wartości w kolumnach Sp_x1, Sp_x2, Sp_x3, Sp_x4 -> działa po prawnie po przebudowie
    for key in Sp_x.keys():
        column_name = f'Sp_{key[-2:]}'
        df_T3[column_name] = float(Sp_x[key]) * df_T3[f'Sp_r_{key[-2:]}']

    # Dodawanie kolumn `F_co_` według wzoru `F_co_x1 = Inc_reve / Sp_x1` -> sprawdzone działa
    for key in Sp_x.keys():
        column_name = f'Sp_{key[-2:]}'
        df_T3[f'F_co_{key[-2:]}'] = df_T3['Inc_reve'] / df_T3[column_name]

    # Dodawanie brakującej kolumny `Sales` z wartościami początkowymi -> działa poprawnie
    df_T3['Sales'] = df_T3['Base_S']
    for sp_key in Sp_x.keys():
        df_T3['Sales'] += df_T3[f'Sp_{sp_key[-2:]}'] * df_T3[f'F_co_{sp_key[-2:]}']

    # Dodawanie kolumn `DMA1_Sp_x1` zgodnie ze słownikiem DMA_dict -> działa poprawnie
    reve_rate_copy = DMA_reve_rate.copy() 
    for sp_key in Sp_x.keys():
        for dma_key in DMA_dict.keys(): 
            if len(reve_rate_copy) == 0: 
                reve_rate_copy = DMA_reve_rate.copy() # Reset listy, jeśli wszystkie wartości zostały wykorzystane 
            DMA_distribution_factor = random_select(reve_rate_copy) 
            df_T3[f'{dma_key}_{sp_key}'] = DMA_distribution_factor * df_T3[f'Sp_{sp_key[-2:]}']

    # Dodawanie kolumn `DMA1_R_co_x1` zgodnie ze słownikami -> działa 
    for sp_key in Sp_x.keys(): 
        for dma_key in DMA_dict.keys(): 
            df_T3[f'{dma_key}_R_co_{sp_key[-2:]}'] = df_T3[f'{dma_key}_Inc_rev'] / df_T3[f'{dma_key}_{sp_key}']

    # Dodawanie kolumn `Sales_DMA` dla każdego DMA
    for dma_key in DMA_dict.keys():
        df_T3[f'Sales_{dma_key}'] = df_T3[f'{dma_key}_BS']
        for sp_key in Sp_x.keys():
            first_calculation = df_T3[f'F_co_{sp_key[-2:]}'] * df_T3[f'{dma_key}_Sp_{sp_key[-2:]}']
            second_calculation = df_T3[f'{dma_key}_R_co_{sp_key[-2:]}'] * df_T3[f'{dma_key}_Sp_{sp_key[-2:]}']
            df_T3[f'Sales_{dma_key}'] += first_calculation + second_calculation   

    # Zapis do pliku Excel
    df_T3.to_excel('Data_T3.xlsx', index=True)
    zamien_nazwy_wierszy(df_T3)
    #return df_T31.T

def zamien_nazwy_wierszy(df_T31):
    global df_T4
    # Funkcja pomocnicza do zamiany końcówek nazw kolumn
    def zamien_koncowke(nazwa):
        nazwa = str(nazwa)  # Konwersja na string, aby uniknąć błędu
        for key, value in marketing_dict.items():
            if key in nazwa:
                return nazwa.replace(key, value)
        return nazwa

    # Funkcja pomocnicza do zamiany prefiksów nazw kolumn
    def zamien_prefiks(nazwa):
        nazwa = str(nazwa)  # Konwersja na string, aby uniknąć błędu
        for key, value in DMA_dict.items():
            if key in nazwa:
                return nazwa.replace(key, value)
        return nazwa

    # Zamiana końcówek nazw kolumn
    df_T4 = df_T31.rename(columns=zamien_koncowke)
    df_T4 = df_T4.rename(columns=zamien_prefiks)
    df_T4 = df_T4.loc[:, ~df_T4.columns.str.startswith('Unnamed')]
    df_T4.to_excel('Data_T4.xlsx', index=True)
    
    return df_T4

# Definicja układu strony
st.title('Marketing Mix Modeling Simulation Engine v2.5')
# Funkcja aktualizująca słownik
def update_dict(old_key, new_key):
    if old_key in marketing_dict:
        marketing_dict[new_key] = marketing_dict.pop(old_key)

# Wyświetlanie formularza
st.subheader('Konfiguracja Parametrów', divider='blue')
col1, col2 = st.columns([0.5, 0.5])

with col1:
    base_sales_total = st.number_input('Base total Sales', value=150000000, key="<k14>")
    periods = st.slider('Data for x periods?', 1, 1, 156, key="<rsi_window>")
    st.write('Ustawiamy sezonowość')
    amplitude = st.slider('Desire function amplitude?', 1, 1, 200, key = "<comm2>")
    mean = st.slider('Desire function mean?', 1, 1, 200, key = "<comm3>")
    std_dev = st.slider('Desire function std_dev?', 1, 1, 200, key = "<comm4>")
    df_sezon = pd.DataFrame(generuj_sezonowosc(periods, amplitude, mean, std_dev))
    df_sezon = df_sezon.rename(columns={0: 'Forecast'})
    df_sezon['Time Period'] = range(1, periods + 1)
    fig_ = px.line(df_sezon, x='Time Period', y=['Forecast'], color_discrete_map={
                    'Forecast': '#636EFA'}, width=1000, height=300) 
    fig_.update_layout(xaxis_title='Time Period', yaxis_title='Values', title='Sales Rate Forecast by Gausse')    
    st.plotly_chart(fig_)
    st.write('Ustawiamy Incentive Revenu Rate')
    amplitude = st.slider('Desire function amplitude?', min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="<com2>")
    frequency = st.slider('Desire function frequency?', min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="<com3>")
    noise_level = st.slider('Desire function noise_level?', min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="<com4>")
    df_inc_rev_rate = pd.DataFrame(generuj_inc_rev_rate(periods, amplitude, frequency, noise_level))
    df_inc_rev_rate = df_inc_rev_rate.rename(columns={0: 'Forecast'})
    df_inc_rev_rate['Time Period'] = range(1, periods + 1)   
    fig_1 = px.line(df_inc_rev_rate, x='Time Period', y=['Forecast'], color_discrete_map={
                    'Forecast': '#FF8C00'}, width=1000, height=300) 
    fig_1.update_layout(xaxis_title='Time Period', yaxis_title='Values', title='Incentive Revenu Rate Forecast by X-Function')    
    st.plotly_chart(fig_1)
    
with col2:
    st.write('Marketing Initiatives')
    df_marketing_dict = pd.DataFrame(marketing_dict, index=[0])
    edited_df_marketing_dict = st.data_editor(df_marketing_dict)
    marketing_dict = edited_df_marketing_dict.iloc[0].to_dict()
    st.write('Updated Marketing Initiatives List')
    st.write(marketing_dict)
    st.write('Marketing Budget Frame')
    df_spx = pd.DataFrame(Sp_x, index=[0])
    edited_df_spx = st.data_editor(df_spx)
    Sp_x = edited_df_spx.iloc[0].to_dict()
    st.write('Updated Marketing Budget Frame')
    st.write(Sp_x)
    st.write('DMA List')
    df_DMA_dict = pd.DataFrame(DMA_dict, index=[0])
    edited_df_DMA_dict = st.data_editor(df_DMA_dict)
    DMA_dict = edited_df_DMA_dict.iloc[0].to_dict()
    st.write('Updated DMA List')
    st.write(DMA_dict)
    
#submitted = st.form_submit_button("Run Symulation")

st.subheader('Dane symulacyjne', divider='red')

if st.button("Run Symulation", type="primary"):
    Data_T3(base_sales_total, periods, Sp_x, marketing_dict, DMA_dict)
    st.subheader('Wizualizacja Symulacji na poziomie Y', divider='red')
    df_T4_s = pd.read_excel('Data_T4.xlsx', index_col=0)
    sp_columns = [col for col in df_T4_s.columns if col.startswith('Sp_')]
    sp_columns = sp_columns[:5]
    y_columns = ['Base_S', 'Inc_reve', 'Sales'] + sp_columns
    
    base_color_map = {'Base_S': '#636EFA', 'Inc_reve': '#00CC96', 'Sales': '#FFD700',
        'Sp_TV': '#EF553B', 'Sp_Facebook': '#AB63FA', 'Sp_Onet': '#FFA15A', 'Sp_Wp': '#19D3F3', 'Sp_Twiter': '#F6BE00'}
    
    new_sp_colors = {sp_columns[i]: list(base_color_map.values())[3 + i % (len(base_color_map) - 3)] for i in range(len(sp_columns))}
    color_discrete_map = {**base_color_map, **new_sp_colors}
    fig_base = px.area(df_T4_s, x='Time Period', y=y_columns, color_discrete_map=color_discrete_map, width=2000, height=600)
    fig_base.update_layout(xaxis_title='Time Period', yaxis_title='Values', title='Yi, Base_S & Marketing Spendings')
    fig_base.update_layout(showlegend=True)
    
    st.plotly_chart(fig_base)

show_table = st.checkbox('Czy chcesz zobaczyć tabele z danymi?')
if show_table:
    df_T4_s1 = pd.read_excel('Data_T4.xlsx', index_col=0)
    for_df = df_T4_s1.T.applymap(lambda x: f"{float(x):,.2f}" if isinstance(x, (int, float)) else x)
    st.markdown(for_df.to_html(escape=False, index=True), unsafe_allow_html=True)

# Analiza DMA
st.subheader('DMA Chart', divider='red')
DMA = st.radio('', list(DMA_dict.values()), horizontal=True)
if st.button("Run DMA Chart"):    
    # Tworzenie tabeli wejściowej na podstawie wybranego miasta
    df_T4_s2 = pd.read_excel('Data_T4.xlsx', index_col=0)
    df_T4_DMA = df_T4_s2[['Time Period', f'{DMA}_BS', f'{DMA}_Inc_rev', f'Sales_{DMA}'] + [col for col in df_T4_s2.columns if col.startswith(f'{DMA}_Sp_')]]  
    # Zidentyfikuj kolumny, które mają prefiks "Sp_"
    sp_columns = [col for col in df_T4_DMA.columns if col.startswith(f'{DMA}_Sp_')]   
    # Ogranicz liczbę kolumn Sp_ do 4
    sp_columns = sp_columns[:5]   
    # Zbierz wszystkie kolumny do wykresu
    y_columns = [f'{DMA}_BS', f'{DMA}_Inc_rev', f'Sales_{DMA}'] + sp_columns  
    # Zdefiniuj mapę kolorów bazując na poprzednich
    base_color_map = {
                f'{DMA}_BS': '#1f77b4',  # niebieski
                f'{DMA}_Inc_rev': '#ff7f0e',  # pomarańczowy
                f'Sales_{DMA}': '#2ca02c',  # zielony
                f'{DMA}_Sp_TV': '#d62728',  # czerwony
                f'{DMA}_Sp_Facebook': '#9467bd',  # fioletowy
                f'{DMA}_Sp_Onet': '#8c564b',  # brązowy
                f'{DMA}_Sp_Wp': '#e377c2', # różowy
                f'{DMA}_Sp_Twiter': '#F6BE00'}        
    # Przypisz kolory do nowych zmiennych "Sp_" bazując na istniejących kolorach
    new_sp_colors = {sp_columns[i]: list(base_color_map.values())[3 + i % (len(base_color_map) - 3)] for i in range(len(sp_columns))}    
    # Połącz mapy kolorów
    color_discrete_map = {**base_color_map, **new_sp_colors}   
    # Tworzenie wykresu
    fig_DMA = px.area(df_T4_DMA, x='Time Period', y=y_columns, color_discrete_map=color_discrete_map, width=2000, height=600)
    fig_DMA.update_layout(xaxis_title='Time Period', yaxis_title='Values', title=f'Analiza {DMA}: Yi, Base_S & Marketing Spendings for {DMA}')
    fig_DMA.update_layout(showlegend=True)    
    st.plotly_chart(fig_DMA)
