import pandas as pd
import numpy as np

def process_countries_data(input_path="data/data_final.csv", 
                           countries_output_path="data/data_countries.csv", 
                           world_output_path="data/data_world.csv"):
    """
    Traite les données pour filtrer uniquement les 195 pays officiels,
    ajoute une colonne continent et sépare les agrégats.
    
    Returns:
        tuple: (data_countries, data_world)
    """
    # Lecture du fichier CSV
    data = pd.read_csv(input_path)

    # Liste officielle des 195 pays
    official_countries = [
        "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
        "The Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi",
        "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo, Democratic Republic of the", "Congo, Republic of the", "Costa Rica", "Côte d'Ivoire", "Croatia", "Cuba", "Cyprus", "Czechia",
        "Denmark", "Djibouti", "Dominica", "Dominican Republic", "East Timor (Timor-Leste)", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia",
        "Fiji", "Finland", "France", "Gabon", "The Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
        "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",
        "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Korea, North", "Korea, South", "Kosovo", "Kuwait", "Kyrgyzstan",
        "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg",
        "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia, Federated States of", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar",
        "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Macedonia", "Norway",
        "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal",
        "Qatar", "Romania", "Russia", "Rwanda",
        "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria",
        "Taiwan, Province of China", "Tajikistan", "Tanzania", "Thailand", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu",
        "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan",
        "Vanuatu", "Holy See", "Venezuela", "Viet Nam", "Yemen", "Zambia", "Zimbabwe"
    ]
    
    # Dictionnaire de correspondance des noms de pays
    country_name_mapping = {
        "Bahamas, The": "The Bahamas",
        "Gambia, The": "The Gambia",
        "Cote d'Ivoire": "Côte d'Ivoire",
        "Congo, Dem. Rep.": "Congo, Democratic Republic of the",
        "Congo, Rep.": "Congo, Republic of the",
        "Iran, Islamic Rep.": "Iran",
        "Korea, Dem. People's Rep.": "Korea, North",
        "Korea, Rep.": "Korea, South",
        "Brunei Darussalam": "Brunei",
        "Kyrgyz Republic": "Kyrgyzstan",
        "Lao PDR": "Laos",
        "Russian Federation": "Russia",
        "Syrian Arab Republic": "Syria",
        "Egypt, Arab Rep.": "Egypt",
        "Turkiye": "Turkey",
        "Yemen, Rep.": "Yemen",
        "Timor-Leste": "East Timor (Timor-Leste)",
        "Venezuela, RB": "Venezuela",
        "Slovak Republic": "Slovakia",
        "Micronesia, Fed. Sts.": "Micronesia, Federated States of",
        "St. Lucia": "Saint Lucia",
        "St. Kitts and Nevis": "Saint Kitts and Nevis",
        "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
        "Vietnam, Socialist Republic of": "Vietnam",
        "Holy See": "Vatican City",
        "Taiwan, Province of China": "Taiwan"
    }

    # Dictionnaire de correspondance entre pays et continents
    continent_mapping = {
        # Afrique
        "Algeria": "Africa", "Angola": "Africa", "Benin": "Africa", "Botswana": "Africa", "Burkina Faso": "Africa", 
        "Burundi": "Africa", "Cabo Verde": "Africa", "Cameroon": "Africa", "Central African Republic": "Africa", 
        "Chad": "Africa", "Comoros": "Africa", "Congo, Democratic Republic of the": "Africa", 
        "Congo, Republic of the": "Africa", "Côte d'Ivoire": "Africa", "Djibouti": "Africa", "Egypt": "Africa", 
        "Equatorial Guinea": "Africa", "Eritrea": "Africa", "Eswatini": "Africa", "Ethiopia": "Africa", 
        "Gabon": "Africa", "The Gambia": "Africa", "Ghana": "Africa", "Guinea": "Africa", "Guinea-Bissau": "Africa", 
        "Kenya": "Africa", "Lesotho": "Africa", "Liberia": "Africa", "Libya": "Africa", "Madagascar": "Africa", 
        "Malawi": "Africa", "Mali": "Africa", "Mauritania": "Africa", "Mauritius": "Africa", "Morocco": "Africa", 
        "Mozambique": "Africa", "Namibia": "Africa", "Niger": "Africa", "Nigeria": "Africa", "Rwanda": "Africa", 
        "Sao Tome and Principe": "Africa", "Senegal": "Africa", "Seychelles": "Africa", "Sierra Leone": "Africa", 
        "Somalia": "Africa", "South Africa": "Africa", "South Sudan": "Africa", "Sudan": "Africa", "Tanzania": "Africa", 
        "Togo": "Africa", "Tunisia": "Africa", "Uganda": "Africa", "Zambia": "Africa", "Zimbabwe": "Africa",
        
        # Amérique du Nord
        "Antigua and Barbuda": "North America", "The Bahamas": "North America", "Barbados": "North America", 
        "Belize": "North America", "Canada": "North America", "Costa Rica": "North America", "Cuba": "North America", 
        "Dominica": "North America", "Dominican Republic": "North America", "El Salvador": "North America", 
        "Grenada": "North America", "Guatemala": "North America", "Haiti": "North America", "Honduras": "North America", 
        "Jamaica": "North America", "Mexico": "North America", "Nicaragua": "North America", "Panama": "North America", 
        "Saint Kitts and Nevis": "North America", "Saint Lucia": "North America", 
        "Saint Vincent and the Grenadines": "North America", "Trinidad and Tobago": "North America", 
        "United States": "North America",
        
        # Amérique du Sud
        "Argentina": "South America", "Bolivia": "South America", "Brazil": "South America", "Chile": "South America", 
        "Colombia": "South America", "Ecuador": "South America", "Guyana": "South America", "Paraguay": "South America", 
        "Peru": "South America", "Suriname": "South America", "Uruguay": "South America", "Venezuela": "South America",
        
        # Asie
        "Afghanistan": "Asia", "Armenia": "Asia", "Azerbaijan": "Asia", "Bahrain": "Asia", "Bangladesh": "Asia", 
        "Bhutan": "Asia", "Brunei": "Asia", "Cambodia": "Asia", "China": "Asia", "Cyprus": "Asia", 
        "East Timor (Timor-Leste)": "Asia", "Georgia": "Asia", "India": "Asia", "Indonesia": "Asia", "Iran": "Asia", 
        "Iraq": "Asia", "Israel": "Asia", "Japan": "Asia", "Jordan": "Asia", "Kazakhstan": "Asia", 
        "Korea, North": "Asia", "Korea, South": "Asia", "Kuwait": "Asia", "Kyrgyzstan": "Asia", "Laos": "Asia", 
        "Lebanon": "Asia", "Malaysia": "Asia", "Maldives": "Asia", "Mongolia": "Asia", "Myanmar": "Asia", 
        "Nepal": "Asia", "Oman": "Asia", "Pakistan": "Asia", "Palestine": "Asia", "Philippines": "Asia", 
        "Qatar": "Asia", "Russia": "Asia", "Saudi Arabia": "Asia", "Singapore": "Asia", "Sri Lanka": "Asia", 
        "Syria": "Asia", "Taiwan, Province of China": "Asia", "Tajikistan": "Asia", "Thailand": "Asia", 
        "Turkey": "Asia", "Turkmenistan": "Asia", "United Arab Emirates": "Asia", "Uzbekistan": "Asia", 
        "Viet Nam": "Asia", "Yemen": "Asia",
        
        # Europe
        "Albania": "Europe", "Andorra": "Europe", "Austria": "Europe", "Belarus": "Europe", "Belgium": "Europe", 
        "Bosnia and Herzegovina": "Europe", "Bulgaria": "Europe", "Croatia": "Europe", "Czechia": "Europe", 
        "Denmark": "Europe", "Estonia": "Europe", "Finland": "Europe", "France": "Europe", "Germany": "Europe", 
        "Greece": "Europe", "Holy See": "Europe", "Hungary": "Europe", "Iceland": "Europe", "Ireland": "Europe", 
        "Italy": "Europe", "Kosovo": "Europe", "Latvia": "Europe", "Liechtenstein": "Europe", "Lithuania": "Europe", 
        "Luxembourg": "Europe", "Malta": "Europe", "Moldova": "Europe", "Monaco": "Europe", "Montenegro": "Europe", 
        "Netherlands": "Europe", "North Macedonia": "Europe", "Norway": "Europe", "Poland": "Europe", 
        "Portugal": "Europe", "Romania": "Europe", "San Marino": "Europe", "Serbia": "Europe", "Slovakia": "Europe", 
        "Slovenia": "Europe", "Spain": "Europe", "Sweden": "Europe", "Switzerland": "Europe", "Ukraine": "Europe", 
        "United Kingdom": "Europe", "Vatican City": "Europe",
        
        # Océanie
        "Australia": "Oceania", "Fiji": "Oceania", "Kiribati": "Oceania", "Marshall Islands": "Oceania", 
        "Micronesia, Federated States of": "Oceania", "Nauru": "Oceania", "New Zealand": "Oceania", 
        "Palau": "Oceania", "Papua New Guinea": "Oceania", "Samoa": "Oceania", "Solomon Islands": "Oceania", 
        "Tonga": "Oceania", "Tuvalu": "Oceania", "Vanuatu": "Oceania"
    }

    # Remplacement des noms de pays
    data['Standard Country Name'] = data['Country Name'].replace(country_name_mapping)
    
    # Filtrer les 195 pays officiels
    data_countries = data[data['Standard Country Name'].isin(official_countries)].copy()
    
    # Ajouter la colonne continent
    data_countries['Continent'] = data_countries['Standard Country Name'].map(continent_mapping)
    
    # Séparer les données mondiales
    data_world = data[data['Country Name'] == 'World'].copy()
    
    # Sauvegarde
    data_countries.to_csv(countries_output_path, index=False)
    data_world.to_csv(world_output_path, index=False)
    
    # Vérification des résultats
    unique_countries = data_countries['Standard Country Name'].nunique()
    print(f"Nombre de pays uniques dans data_countries : {unique_countries}")
    
    missing = set(official_countries) - set(data_countries['Standard Country Name'].unique())
    if missing:
        print("Pays manquants :", missing)
    
    return data_countries, data_world

if __name__ == "__main__":
    process_countries_data()

