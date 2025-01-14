import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import random
from datetime import datetime, date

# Ustawienia strony
st.set_page_config(layout="wide")
marketing_dict = {'_x1': '_TV', '_x2': '_Facebook', '_x3': '_Onet', '_x4': '_Wp','_x5':'_GW'}
Sp_x = {'Sp_x1':'35000000','Sp_x2':'25000000','Sp_x3':'15000000','Sp_x4':'10000000','Sp_x5':'15000000'}
DMA_dict = {'DMA1':'Warszawa','DMA2':'Kraków','DMA3':'Poznań','DMA4':'Gdańsk', 'DMA5':'Katowice'}
DMA_reve_rate = [0.18, 0,13, 0.27, 0.19, 0.23] #[0.2, 0.15, 0.3, 0.1, 0.25]  # golden key :-)

# Generowanie Incentive Revenue Rate krzywą Gaussa z szumem
def generuj_inc_rev_rate(periods, amplitude, frequency, noise_level):
    czas = np.arange(periods)
        #smooth_noise = np.cumsum(np.random.normal(0, noise_level, periods))  # Ten szum jest za duży
    smooth_noise = amplitude * np.sin(2 * np.pi * frequency * czas / periods) + noise_level * np.sin(4 * np.pi * frequency * czas / periods)
        #smooth_noise = amplitude * np.arctan(2 * np.pi * frequency * czas / periods) + noise_level * np.arctan(4 * np.pi * frequency * czas / periods)
        #smooth_noise = amplitude * np.arctan(np.sin(2 * np.pi * frequency * czas / periods) + noise_level * np.sin(4 * np.pi * frequency * czas / periods))
    smooth_noise = np.clip(smooth_noise, -0.15 * amplitude, None)
    inc_rev_rate = amplitude * (np.sin(czas / 10 * frequency) + np.cos(czas / 5 * frequency)) + smooth_noise
    inc_rev_rate = (inc_rev_rate - np.min(inc_rev_rate)) / (np.max(inc_rev_rate) - np.min(inc_rev_rate)) * 2 # skalowanie do 2
        #inc_rev_rate = np.clip(inc_rev_rate, -0.1 * amplitude, None)
        #inc_rev_rate = amplitude * (np.arctan(czas / 10 * frequency) + np.sin(czas / 5 * frequency)) + smooth_noise
        #inc_rev_rate = (inc_rev_rate - np.min(inc_rev_rate)) / (np.max(inc_rev_rate) - np.min(inc_rev_rate))  # Skaluje do zakresu [0, 1]
        #inc_rev_rate = inc_rev_rate / np.sum(inc_rev_rate)  # Normalizuje, aby suma wynosiła 1
    return inc_rev_rate

# Generowanie sezonowości w oparciu o krzywą Gaussa
def generuj_sezonowosc(periods, amplitudeT, mean, std_dev):
    global sezonowosc
    czas = np.arange(periods)
    sezonowosc = amplitudeT * np.exp(-0.5 * ((czas - mean) / std_dev) ** 2)
    sezonowosc = (sezonowosc - np.min(sezonowosc)) / (np.max(sezonowosc) - np.min(sezonowosc))  # Skaluje do zakresu [0, 1]
    sezonowosc = sezonowosc / np.sum(sezonowosc)  # Normalizuje, aby suma wynosiła 1
    return sezonowosc

# Generowanie sezonowości w oparciu o krzywą Gaussa
def generuj_sp(periods, amplitudeP, meanP, std_devP):
    global Spending_rate
    czas = np.arange(periods)
    Spending_rate = amplitudeP * np.exp(-0.5 * ((czas - meanP) / std_devP) ** 2)
    Spending_rate = (Spending_rate - np.min(Spending_rate)) / (np.max(Spending_rate) - np.min(Spending_rate))  # Skaluje do zakresu [0, 1]
    Spending_rate = Spending_rate / np.sum(Spending_rate)  # Normalizuje, aby suma wynosiła 1
    
    return Spending_rate

def random_select(reve_rate_list):
    selected_value = random.choice(reve_rate_list)
    reve_rate_list.remove(selected_value)
    return selected_value

def Data_T3(base_sales_total, periods, df_sezon, df_inc_rev_rate, df_spending_rate, Sp_x, marketing_dict, DMA_dict):
    global df_T3
       
    # Tworzenie DataFrame z kolumnami
    df_T3 = pd.DataFrame({
        'Time Period': [f'P{i+1}' for i in range(periods)],
        'Base_S_Plan_rate': df_sezon,
        'Base_S': df_sezon * base_sales_total, # sezonowosc
        'Inc_rev_rate': df_inc_rev_rate 
    })
    # Obliczanie wartości w kolumnie 'Inc_reve' jako iloczyn 'Base_S' i 'Inc_rev_rate' 
    df_T3['Inc_reve'] = df_T3['Base_S'] * df_T3['Inc_rev_rate']
    
    # Obliczanie sumy wartości w słowniku
    sum_sp_x_dict = sum(float(value) for value in Sp_x.values())

    # Obliczanie udziału procentowego w formie ułamka dziesiętnego
    sales_share = {key: float(value) / sum_sp_x_dict for key, value in Sp_x.items()}

    # Tworzenie df_sales_share
    df_sales_share = pd.DataFrame(list(sales_share.items()), columns=['Sa_r_x', 'Value'])
    df_sales_share['Sa_r_x'] = [f'Sa_r_x{index+1}' for index in range(len(Sp_x))]
    df_sales_share['Value'] = df_sales_share['Value'].map('{:.3f}'.format)
  
    # Obliczanie sumy wartości w słowniku Sp_x -> wyniki ok
    sum_sp_x_dict = sum(float(value) for value in Sp_x.values())
    st.write(f"Total Marketing Spendings: {sum_sp_x_dict:,.2f}")
    #print(f"Suma wartości w słowniku Sp_x: {sum_sp_x_dict:,.2f}")
    # Obliczanie sumy kolumny Inc_reve > wyniki ok
    sum_inc_reve = df_T3['Inc_reve'].sum()
    st.write(f"Total Incentive Revenue: {sum_inc_reve:,.2f}")
    #print(f"Suma kolumny Inc_reve: {sum_inc_reve:,.2f}")

    # Dodawanie kolumn `DMA1_BS` zgodnie ze słownikiem DMA_dict -> to zostało przebudowane i działą poprawnie
    reve_rate_copy = DMA_reve_rate.copy()
    for dma_key in DMA_dict.keys():
        if len(reve_rate_copy) == 0:
            reve_rate_copy = DMA_reve_rate.copy()  # Reset listy, jeśli wszystkie wartości zostały wykorzystane
        DMA_distribution_factor = random_select(reve_rate_copy)
        df_T3[f'{dma_key}_BS'] = DMA_distribution_factor * df_T3['Base_S']

    # Dodawanie brakujących kolumn `Sp_r_x` -> to działa jak szalone -> nowa funkcja
    for key in Sp_x.keys():
        column_name = f'Sp_{key[-2:]}'
        df_T3[f'Sp_r_{key[-2:]}'] = df_spending_rate

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
        df_T3[f'F_co_{key[-2:]}'] = (df_T3['Inc_reve'] / df_T3[column_name]) # * (1 + df_inc_rev_rate)**2  # tutaj dodałem boosta

    # Dodawanie brakującej kolumny `Sales` z wartościami początkowymi
    df_T3['Sales'] = df_T3['Base_S']
    # Iterowanie przez klucze w Sp_x i akumulowanie wartości
    for sp_key in Sp_x.keys():
        df_T3['Sales'] += df_T3[f'Sp_{sp_key[-2:]}'] * df_T3[f'F_co_{sp_key[-2:]}']

     # Tworzenie kolumn Sa_x w df_T3
    for index, row in df_sales_share.iterrows():
        column_name = row['Sa_r_x'].replace('Sa_r_', 'Sa_')
        df_T3[column_name] = float(row['Value']) * df_T3['Sales']
    
    # Dodawanie brakujących kolumn typu `DMA1_Sp_r_x1` i wypełnianie danymi z df_spending_rate 
    for dma_key in DMA_dict.keys(): 
        for key in Sp_x.keys(): 
            column_name = f'{dma_key}_Sp_r_{key[-2:]}' 
            df_T3[column_name] = df_spending_rate

    # Dodanie kolumn DMA_Sp_x jako iloczyn odpowiednich kolumn DMA_Sp_r_x i Sp_x
    for key in Sp_x.keys():
        for dma_key in DMA_dict.keys():
            column_name = f'Sp_{key[-2:]}'
            dma_column_name = f'{dma_key}_{column_name}'
            df_T3[dma_column_name] = df_T3[f'{dma_key}_Sp_r_{key[-2:]}'] * df_T3[column_name]

    # Dodawanie kolumn `DMA1_R_co_x1` zgodnie ze słownikami -> działa 
    for sp_key in Sp_x.keys(): 
        for dma_key in DMA_dict.keys(): 
            df_T3[f'{dma_key}_R_co_{sp_key[-2:]}'] = df_T3[f'{dma_key}_Inc_rev'] / df_T3[f'{dma_key}_{sp_key}']

    # Dodawanie kolumn `Sales_DMA` dla każdego DMA
    for dma_key in DMA_dict.keys():
        df_T3[f'Sales_{dma_key}'] = df_T3[f'{dma_key}_BS']
        for sp_key in Sp_x.keys():
            first_calculation = df_T3[f'F_co_{sp_key[-2:]}'].mean() * df_T3[f'{dma_key}_Sp_{sp_key[-2:]}'] # zdjąłem średnią na F_co .mean()
            second_calculation = df_T3[f'{dma_key}_R_co_{sp_key[-2:]}'] * df_T3[f'{dma_key}_Sp_{sp_key[-2:]}']
            df_T3[f'Sales_{dma_key}'] += (first_calculation + second_calculation)

    # Dodawanie kolumn `DMA_Sa_x` dla każdego DMA
    for dma_key in DMA_dict.keys():
        for index, row in df_sales_share.iterrows():
            column_name = f'DMA_{dma_key}_{row["Sa_r_x"].replace("Sa_r_", "Sa_")}'
            df_T3[column_name] = float(row['Value']) * df_T3[f'Sales_{dma_key}']
        
    # Zapis do pliku Excel
    df_T3 = df_T3.replace([np.nan, np.inf, -np.inf], 0)
    df_T3.to_excel('Data_T3.xlsx', index=True)
    zamien_nazwy_wierszy(df_T3)
    
def zamien_nazwy_wierszy(df_T3):
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
    df_T4 = df_T3.rename(columns=zamien_koncowke)
    df_T4 = df_T4.rename(columns=zamien_prefiks)
    df_T4 = df_T4.loc[:, ~df_T4.columns.str.startswith('Unnamed')]
    df_T4 = df_T4.replace([np.nan, np.inf, -np.inf], 0)
    df_T4.to_excel('Data_T4.xlsx', index=True)
      
    return df_T4

# Definicja układu strony
st.title('Marketing Mix Modeling Simulation Dashboard v3 ')
# Styl zakładki bocznej
st.html("""<style>[data-testid="stSidebarContent"] {color: black; background-color: #009A17} </style>""")
st.sidebar.subheader('Choose an analytical tool') 

# Funkcja aktualizująca słownik
def update_dict(old_key, new_key):
    if old_key in marketing_dict:
        marketing_dict[new_key] = marketing_dict.pop(old_key)
         
def Seasonality_Trend(periods, amplitudeT, mean, std_dev ):   
    global df_sezon
    df_sezon = np.array(generuj_sezonowosc(periods, amplitudeT, mean, std_dev))
    df_sezon_df = pd.DataFrame(df_sezon)
    df_sezon_df = df_sezon_df.rename(columns={0: 'Forecast'})
    df_sezon_df['Time Period'] = range(1, periods + 1)
    fig_ = px.line(df_sezon_df, x='Time Period', y=['Forecast'], color_discrete_map={
                    'Forecast': '#636EFA'}, width=1000, height=300) 
    fig_.update_layout(xaxis_title='Time Period', yaxis_title='Values', title='Sales Rate Forecast by Gausse')    
    st.plotly_chart(fig_)

def Incentive_Revenu_Rate_Function(periods, amplitude, frequency, noise_level ):
    global df_inc_rev_rate
    df_inc_rev_rate = np.array(generuj_inc_rev_rate(periods, amplitude, frequency, noise_level))
    df_inc_rev_rate_df = pd.DataFrame(df_inc_rev_rate)
    df_inc_rev_rate_df = df_inc_rev_rate_df.rename(columns={0: 'Forecast'})
    df_inc_rev_rate_df['Time Period'] = range(1, periods + 1)   
    fig_1 = px.line(df_inc_rev_rate_df, x='Time Period', y=['Forecast'], color_discrete_map={
                    'Forecast': '#FF8C00'}, width=1000, height=300) 
    fig_1.update_layout(xaxis_title='Time Period', yaxis_title='Values', title='Incentive Revenu Rate Forecast by X-Function')    
    st.plotly_chart(fig_1)

def Spending_Rate_Function(periods, amplitudeP, frequencyP, noise_levelP):
    global df_spending_rate
    df_spending_rate = np.array(generuj_sp(periods, amplitudeP, meanP, std_devP))
    df_spending_rate_df = pd.DataFrame(df_spending_rate)
    df_spending_rate_df = df_spending_rate_df.rename(columns={0: 'Forecast'})
    df_spending_rate_df['Time Period'] = range(1, periods + 1) 
    fig_2 = px.line(df_spending_rate_df, x='Time Period', y=['Forecast'], color_discrete_map={
                    'Forecast': '#6B8E23'}, width=1000, height=300) 
    fig_2.update_layout(xaxis_title='Time Period', yaxis_title='Values', title='Spending Rate Forecast by Gausse')    
    st.plotly_chart(fig_2)

st.subheader('Data Symulation', divider='red')
col1, col2 = st.columns([0.5, 0.5])
with col1:
    st.write('Principle Parameters')
    base_sales_total = st.number_input('Base total Sales', value=400000000, key="<k14>")
    periods = st.slider('Data for x periods?', 1, 1, 156, key="<rsi_window>")
    
    st.write('Seasonality Trend Function')
    amplitudeT = st.slider('Desire function amplitude?', 1, 1, 200, key = "<comm2>")
    mean = st.slider('Desire function mean?', 1, 1, 200, key = "<comm3>")
    std_dev = st.slider('Desire function std_dev?', 1, 1, 200, key = "<comm4>")
    Seasonality_Trend(periods, amplitudeT, mean,std_dev )
    
    st.write('Incentive Revenu Rate Function')
    amplitude = st.slider('Desire function amplitude?', min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="<com2>")
    frequency = st.slider('Desire function frequency?', min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="<com3>")
    noise_level = st.slider('Desire function noise_level?', min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="<com4>")
    Incentive_Revenu_Rate_Function(periods, amplitude, frequency, noise_level )
    
    st.write('Marketing Spending Trend Function')
    amplitudeP = st.slider('Desire function amplitude?', 1, 1, 200, key = "<comm21>")
    meanP = st.slider('Desire function mean?', 1, 1, 200, key = "<comm31>")
    std_devP = st.slider('Desire function std_dev?', 1, 1, 200, key = "<comm41>")
    Spending_Rate_Function(periods, amplitudeP, meanP, std_devP)  
with col2:
    marketing_dict = {'_x1': '_TV', '_x2': '_Facebook', '_x3': '_Onet', '_x4': '_Wp','_x5':'_GW'}
    Sp_x = {'Sp_x1':'35000000','Sp_x2':'25000000','Sp_x3':'15000000','Sp_x4':'10000000','Sp_x5':'15000000'}
    DMA_dict = {'DMA1':'Warszawa','DMA2':'Kraków','DMA3':'Poznań','DMA4':'Gdańsk', 'DMA5':'Katowice'}
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
        
if st.button("Configure Parameters & Run Simulation"):
    Data_T3(base_sales_total, periods, df_sezon, df_inc_rev_rate, df_spending_rate, Sp_x, marketing_dict, DMA_dict)    

checkbox_table = st.sidebar.checkbox('Do you want to see data table ?', key="<aver1>")
if checkbox_table:
    df_T4_s = pd.read_excel('Data_T4.xlsx', index_col=0)
    for_df = df_T4_s.T.applymap(lambda x: f"{float(x):,.2f}" if isinstance(x, (int, float)) else x)
    st.markdown(for_df.to_html(escape=False, index=True), unsafe_allow_html=True)

# Sekcja wykres Run Sales decomposition chart
def run_sales_decomposition_chart():
    st.subheader('Total Sales decomposition chart', divider='red')
    df_T4_s1 = pd.read_excel('Data_T4.xlsx', index_col=0)
    sp_columns = [col for col in df_T4_s1.columns if col.startswith('Sa_')]
    sp_columns = sp_columns[:5]
    y_columns = ['Sales', 'Base_S'] + sp_columns
    
    column_sums = df_T4_s1[y_columns].sum().sort_values(ascending=True)
    sorted_y_columns = column_sums.index.tolist()
    
    # Tworzenie wykresu typu area
    fig_base = px.area(df_T4_s1, x='Time Period', y=sorted_y_columns, color_discrete_sequence=px.colors.sequential.Viridis, width=1000, height=400)
    fig_base.update_layout(xaxis_title='Time Period', yaxis_title='Values', title='Sales, Base_S & Marketing Spendings', showlegend=True)
    fig_base.update_layout(showlegend=True)
    st.plotly_chart(fig_base)

checkbox_decomp_chart = st.sidebar.checkbox('Run Sales decomposition chart', key="<aver2>")
if checkbox_decomp_chart:
    run_sales_decomposition_chart()
 
# Sekcja DMA Sales Decomposition Chart 
def run_dma_sales_decomposition_chart(DMA):
    df_T4_s2 = pd.read_excel('Data_T4.xlsx', index_col=0)
    
    # Tworzenie pełnej listy kolumn DMA_Sa zgodnie ze słownikami
    sa_columns = [col for col in df_T4_s2.columns if f'{DMA}_Sa_' in col]
    # Tworzenie listy kolumn BS zgodnie z wybranym DMA
    bs_columns = [f'{DMA}_BS']
    # Tworzenie listy kolumn Sales zgodnie z wybranym DMA
    sales_columns = [f'Sales_{DMA}']
    # Łączenie kolumn BS i Sales
    yy_columns = bs_columns + sales_columns + sa_columns
    
    # Sortowanie kolumn według sumy wartości w df_T4_s2
    column_sums = df_T4_s2[yy_columns].sum().sort_values(ascending=True)
    sorted_yy_columns = column_sums.index.tolist()
    
    # Tworzenie wykresu typu area z posortowanymi kolumnami
    fig_DMA = px.area(df_T4_s2, x='Time Period', y=sorted_yy_columns, color_discrete_sequence=px.colors.sequential.Turbo, width=1000, height=400)
    fig_DMA.update_layout(xaxis_title='Time Period', yaxis_title='Values', title=f'Analiza {DMA}: Sales, Base_S & Marketing Spendings for {DMA}')
    fig_DMA.update_layout(showlegend=True)
    st.plotly_chart(fig_DMA)

checkbox_dma_chart = st.sidebar.checkbox('Run DMA Sales decomposition chart', key="<aver3>")
if checkbox_dma_chart:    
    st.subheader('DMA Sales Decomposition Chart', divider='red')
    DMA = st.radio('', list(DMA_dict.values()), horizontal=True, key='DMA_radio')
    run_dma_sales_decomposition_chart(DMA)

# Checkbox Efficency 
checkbox_efficiency_chart = st.sidebar.checkbox('Run Media Efficiency Curves', key="<new_key>")
def run_efficiency_chart():
    # Źródło danych
    df_T4_s3 = pd.read_excel('Data_T4.xlsx', index_col=0)
    # Kolumny z wydatkami na media
    sp_columns = [col for col in df_T4_s3.columns if col.startswith('Sp_') and not 'Sp_r_' in col]  
    # Tworzenie danych do wykresu
    plot_data = pd.DataFrame()
    for sp_col in sp_columns:
        temp_df = pd.DataFrame({
            'Investments': df_T4_s3[sp_col],
            'Sales Results': df_T4_s3['Sales'] / df_T4_s3[sp_col],
            'Media': sp_col
        }).fillna(0)  
        plot_data = pd.concat([plot_data, temp_df])
    
    # Tworzenie wykresu za pomocą plotly express
    fig_eff1 = px.line(plot_data, x='Investments', y='Sales Results', color= 'Media', width=2000, height=600, 
                  title='Media Efficiency Curves')
    
    # Ustawienia osi
    fig_eff1.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=50000, range=[0, 300000], showgrid=True, 
                            gridwidth=1, gridcolor='LightGrey', griddash='dash' ),
            yaxis=dict(tickmode='array', tickvals=[1e5, 1e6, 1e7, 1e8, 1e9], ticktext=['100k', '1M', '10M', '100M', '1B'],
                type='log', showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dash' ), title_x=0.5, template='plotly_white')
    
    st.plotly_chart(fig_eff1)

if checkbox_efficiency_chart:
    run_efficiency_chart()





