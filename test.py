import os                                                                              
import sys                                                                             
import io                                                                              
import matplotlib.pyplot as plt                                                        
import matplotlib.ticker as mtick                                                      
import numpy as np                                                                     
import pandas as pd                                                                    
from dotenv import load_dotenv                                                         
from supabase import create_client, Client                                             
from PIL import Image                                                                  
                                                                                       
try:                                                                                   
    load_dotenv()                                                                      
                                                                                       
    SUPABASE_URL = os.getenv("SUPABASE_URL")                                           
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")                                           
    SUPABASE_BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME")                           
                                                                                       
    if not all([SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET_NAME]):                    
        raise ValueError("Missing environment variables")                              
                                                                                       
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)                       
                                                                                       
    response = supabase.table("8ep9aa0eunw6ge0cws1r").select("*").execute()            
                                                                                       
    if len(response.data) == 0:                                                        
        raise ValueError("No data retrieved from Supabase")                            
                                                                                       
    df = pd.DataFrame(response.data)                                                   
                                                                                       
    gujarat_data = df[df['state_ut'].str.lower().str.contains('gujarat') |             
                      df['state_ut'].str.lower().str.contains('gj')]                   
                                                                                       
    if len(gujarat_data) == 0:                                                         
        industries = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',           
'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']                                           
        max_states = min(5, len(df))                                                   
        sample_data = df.iloc[:max_states]                                             
                                                                                       
        fig, ax = plt.subplots(figsize=(12, 8))                                        
                                                                                       
        for idx, row in sample_data.iterrows():                                        
            state = row['state_ut']                                                    
            e_value = row['e']                                                         
            ax.bar(state, e_value, color=plt.cm.Set3(idx % 12))                        
                                                                                       
        ax.set_ylabel('Percentage (%)')                                                
        ax.set_title('NIC Industrial Class E Workers by State (Gujarat not             \
found in data)')                                                                       
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))                 
        plt.xticks(rotation=45, ha='right')                                            
        plt.tight_layout()                                                             
                                                                                       
        message = "Gujarat data not found in the database. Showing sample              \
data for other states."                                                                
    else:                                                                              
        e_value = gujarat_data['e'].values[0]                                          
                                                                                       
        industries = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',           
'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']                                           
        values = [gujarat_data[col.lower()].values[0] for col in industries]           
                                                                                       
        colors = plt.cm.viridis(np.linspace(0, 1, len(industries)))                    
        highlight_color = 'red'                                                        
                                                                                       
        fig, ax = plt.subplots(figsize=(14, 10))                                       
                                                                                       
        bars = ax.bar(industries, values, color=colors)                                
        bars[4].set_color(highlight_color)                                             
                                                                                       
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)                        
        ax.set_title(f'Distribution of Workers by NIC Industrial Class in              \
{gujarat_data["state_ut"].values[0]}', fontsize=16)                                    
        ax.set_xlabel('NIC Industrial Class', fontsize=12)                             
        ax.set_ylabel('Percentage of Workers (%)', fontsize=12)                        
                                                                                       
        for i, v in enumerate(values):                                                 
            label_color = 'black' if i != 4 else 'white'                               
            ax.text(i, v + 0.5, f'{v:.2f}%', color=label_color, ha='center',           
fontsize=10)                                                                           
                                                                                       
        ax.grid(axis='y', linestyle='--', alpha=0.7)                                   
                                                                                       
        plt.annotate(f'Class E: {e_value:.2f}%',                                       
                    xy=(4, e_value),                                                   
                    xytext=(4, e_value + max(values)/10),                              
                    fontsize=12,                                                       
                    weight='bold',                                                     
                    color=highlight_color,                                             
                    arrowprops=dict(facecolor=highlight_color, shrink=0.05,            
width=2))                                                                              
                                                                                       
        plt.tight_layout()                                                             
        message = f"Percentage of workers in NIC Industrial Class E in {gujarat_data['state_ut'].values[0]}: {e_value:.2f}%"                                  
                                                                                       
    img_data = io.BytesIO()                                                            
    plt.savefig(img_data, format='png', dpi=300, bbox_inches='tight')                  
    img_data.seek(0)                                                                   
                                                                                       
    file_name = 'gujarat_nic_e_workers.png'                                            
                                                                                       
    img = Image.open(img_data)                                                         
    img_buffer = io.BytesIO()                                                          
    img.save(img_buffer, format='PNG')                                                 
    img_buffer.seek(0)                                                                 
                                                                                       
    res = supabase.storage.from_(SUPABASE_BUCKET_NAME).upload(                         
        file_name,                                                                     
        img_buffer,                                                                    
        file_options={"content-type": "image/png"}                                     
    )                                                                                  
                                                                                       
    public_url =                                                                       supabase.storage.from_(SUPABASE_BUCKET_NAME).get_public_url(file_name)                 
    SUPABASE_GRAPH_URL = public_url                                                    
                                                                                       
    plt.close()                                                                        
                                                                                       
except Exception as e:                                                                 
    file_name = 'error_report.png'                                                     
                                                                                       
    fig, ax = plt.subplots(figsize=(10, 6))                                            
    ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center',                    
fontsize=12)                                                                           
    ax.set_axis_off()                                                                  
                                                                                       
    img_data = io.BytesIO()                                                            
    plt.savefig(img_data, format='png', dpi=300)                                       
    img_data.seek(0)                                                                   
                                                                                       
    img = Image.open(img_data)                                                         
    img_buffer = io.BytesIO()                                                          
    img.save(img_buffer, format='PNG')                                                 
    img_buffer.seek(0)                                                                 
                                                                                       
    try:                                                                               
        res = supabase.storage.from_(SUPABASE_BUCKET_NAME).upload(                     
            file_name,                                                                 
            img_buffer,                                                                
            file_options={"content-type": "image/png"}                                 
        )                                                                              
                                                                                       
        public_url =                                                                   
supabase.storage.from_(SUPABASE_BUCKET_NAME).get_public_url(file_name)                 
        SUPABASE_GRAPH_URL = public_url                                                
    except:                                                                            
        SUPABASE_GRAPH_URL = "Error generating visualization"                          
                                                                                       
    plt.close()          
