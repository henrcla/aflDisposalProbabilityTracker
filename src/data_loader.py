import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

BASE_URL = "https://afltables.com"
TEAMS = ["adelaide", "brisbane", "carlton", "collingwood",
    "essendon", "fremantle", "geelong", "goldcoast",
    "gws", "hawthorn", "melbourne", "northmelbourne",
    "portadelaide", "richmond", "stkilda", "sydney",
    "westcoast", "westernbulldogs"]

SEASONS = [2022, 2023, 2024]

RAW_DATA_PATH = "data/raw/"

# Player page URL structure: https://afltables.com/afl/stats/players/[FIRST LETTER OF FIRST NAME]/[First_Last].html
# e.g. Sam Berry -> /S/Sam_Berry.html

# Player page table structure:
# Table 1: Season by season stats (DA column = disposal average)
# Table 4: By opponent (DA per opponent) and by venue (DA per venue)
# Tables 5+: Game by game stats per season

def get_player_url(player_name):
    # Builds afltables url for player stat page
  
    # Replace the space between first and last name with an underscore
    formatted_name = player_name.replace(" ", "_")
    
    # Get the first letter of the first name for the URL folder
    first_letter = player_name[0]
    
    # Build and return the full URL
    url = f"{BASE_URL}/afl/stats/players/{first_letter}/{formatted_name}.html"
    return url

def get_page(url):
    # Fetch webpage
    try:
        # Add a delay to avoid overwhelming the server
        time.sleep(1)
        
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content and return it
            return BeautifulSoup(response.content, "html.parser")
        else:
            print(f"Failed to fetch {url} - Status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def get_all_player_data(player_name):
    """
    Fetches a player's page once and extracts all data we need:
    - Game by game results
    - Career DA vs each opponent  
    - Career DA at each venue
    
    Returns a tuple of (games_df, opponent_stats, venue_stats)
    """
    url = get_player_url(player_name)
    soup = get_page(url)
    
    if soup is None:
        return pd.DataFrame(), {}, {}
    
    try:
        tables = soup.find_all("table")
        
        # --- Extract game by game data ---
        all_games = []
        
        for table in tables[7:]:
            first_row = table.find("tr")
            season_label = first_row.text.strip()
            
            try:
                year = int(season_label.split("-")[-1].strip())
            except ValueError:
                continue
            
            rows = table.find_all("tr")[2:]
            
            for row in rows:
                cells = row.find_all("td")
                
                if len(cells) < 8:
                    continue
                
                opponent = cells[1].text.strip()
                
                if not opponent:
                    continue
                
                round_num = cells[2].text.strip()
                result = cells[3].text.strip()
                di_text = cells[7].text.strip()
                
                if not di_text:
                    continue
                
                disposals = int(di_text)
                pct_text = cells[-1].text.strip()
                game_pct = float(pct_text) if pct_text else None
                
                all_games.append({
                    "player": player_name,
                    "season": year,
                    "opponent": opponent,
                    "round": round_num,
                    "result": result,
                    "disposals": disposals,
                    "game_pct": game_pct
                })
        
        games_df = pd.DataFrame(all_games)
        
        # --- Extract opponent stats ---
        opponent_stats = {}
        opponent_table = tables[5]
        
        for row in opponent_table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) < 6:
                continue
            opponent = cells[0].text.strip()
            if not opponent or opponent == "Totals":
                continue
            da_text = cells[6].text.strip()
            if da_text:
                try:
                    opponent_stats[opponent] = float(da_text)
                except ValueError:
                    continue
        
        # --- Extract venue stats ---
        venue_stats = {}
        venue_table = tables[6]
        
        for row in venue_table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) < 6:
                continue
            venue = cells[0].text.strip()
            if not venue or venue == "Totals":
                continue
            da_text = cells[6].text.strip()
            if da_text:
                try:
                    venue_stats[venue] = float(da_text)
                except ValueError:
                    continue
        
        return games_df, opponent_stats, venue_stats
    
    except Exception as e:
        print(f"Error processing {player_name}: {e}")
        return pd.DataFrame(), {}, {}

# Temporary test
games_df, opponent_stats, venue_stats = get_all_player_data("Sam Berry")