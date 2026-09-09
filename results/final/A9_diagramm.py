import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_summary_files():
    # Sucht nach dem neuen Namensschema, z.B. 20m_strasse1_geraetA_summary.txt
    files = glob.glob("*summary.txt")
    data = []

    for file in files:
        # Regex für das neue Namensschema
        match = re.match(r"^(\d+)m_([a-zA-Z0-9]+)_geraet([AB])_summary\.txt$", file, re.IGNORECASE)
        if not match:
            continue
            
        distanz = int(match.group(1))
        umgebung = match.group(2).capitalize()
        geraet = match.group(3).upper()

        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            ber_match = re.search(r"BER:\s*([0-9.]+)%", content)
            loss_match = re.search(r"\(([-0-9.]+)%\s*Verlust\)", content)
            
            if ber_match and loss_match:
                data.append({
                    "Distanz": distanz,
                    "Umgebung": umgebung,
                    "Gerät": geraet,
                    "BER": float(ber_match.group(1)),
                    "Verlust": float(loss_match.group(1))
                })

    return pd.DataFrame(data)

def plot_tdma_data(df):
    if df.empty:
        print("Keine Daten gefunden. Überprüfe die Dateinamen!")
        return

    # Durchschnittswerte über alle Umgebungen pro Distanz und Gerät
    df_mean = df.groupby(['Distanz', 'Gerät'])[['BER', 'Verlust']].mean().reset_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=False)
    fig.suptitle('SDR TDMA-Feldtests: Signalauswertung und Kanalsymmetrie', fontsize=15, fontweight='bold')

    # --- 1. BER vs Distanz (Getrennt nach Gerät A und B) ---
    for geraet, color, marker in zip(['A', 'B'], ['royalblue', 'firebrick'], ['o', 's']):
        subset = df_mean[df_mean['Gerät'] == geraet].sort_values(by='Distanz')
        if not subset.empty:
            ax1.plot(subset['Distanz'], subset['BER'], marker=marker, linewidth=2, markersize=8, 
                     label=f'Gerät {geraet} (Empfang)', color=color)
            
    ax1.set_title('Entwicklung der Bitfehlerrate (BER) über die Distanz', fontweight='bold')
    ax1.set_xlabel('Distanz [m]')
    ax1.set_ylabel('Ø BER [%]')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # --- 2. Bar-Chart: TDMA Kanalsymmetrie (Gesamtdurchschnitt) ---
    # Zeigt, ob A oder B generell besser empfangen hat
    ber_a = df[df['Gerät'] == 'A']['BER'].mean()
    ber_b = df[df['Gerät'] == 'B']['BER'].mean()
    
    if not pd.isna(ber_a) and not pd.isna(ber_b):
        bars = ax2.bar(['Gerät A', 'Gerät B'], [ber_a, ber_b], color=['royalblue', 'firebrick'], width=0.4)
        ax2.set_title('TDMA Kanalsymmetrie: Durchschnittliche BER über alle Tests', fontweight='bold')
        ax2.set_ylabel('Ø BER [%]')
        
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("TDMA_Resultate.png", dpi=300)
    print("Erfolg! Diagramm gespeichert als 'TDMA_Resultate.png'")
    plt.show()

if __name__ == "__main__":
    df = parse_summary_files()
    plot_tdma_data(df)