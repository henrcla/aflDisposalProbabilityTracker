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

def format_player_name(raw_name):
    """
    Converts "Berry, Sam" format to "Sam Berry" format.
    
    Example:
        "Berry, Sam" -> "Sam Berry"
        "O'Brien, Reilly" -> "Reilly O'Brien"
    """
    # Split on the comma
    parts = raw_name.split(",")
    
    if len(parts) != 2:
        return raw_name
    
    surname = parts[0].strip()
    first_name = parts[1].strip()
    
    return f"{first_name} {surname}"

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

def get_team_players(team, season):
    """
    Scrapes all player names from a team's season page.
    
    Returns a list of player names like:
        ["Sam Berry", "Jordan Dawson", "Rory Laird", ...]
    """
    url = f"{BASE_URL}/afl/stats/teams/{team}/{season}_gbg.html"
    soup = get_page(url)
    
    if soup is None:
        return []
    
    try:
        tables = soup.find_all("table")
        
        # Player stats are in the first table
        player_table = tables[0]
        rows = player_table.find_all("tr")
        
        players = []
        
        for row in rows:
            cells = row.find_all("td")
            
            if not cells:
                continue
            
            # First cell contains the player name as a link
            first_cell = cells[0]
            link = first_cell.find("a")
            
            if link:
                player_name = link.text.strip()
                if player_name:
                    players.append(format_player_name(player_name))
        
        return players
    
    except Exception as e:
        print(f"Error getting players for {team} {season}: {e}")
        return []

def save_all_data():
    """
    Master function that scrapes all player data across all teams and seasons.
    Saves results to data/raw/ as CSV files.
    
    Skips players already scraped to allow resuming if interrupted.
    """
    # Create output directory if it doesn't exist
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    
    all_games = []
    all_opponent_stats = {}
    all_venue_stats = {}
    
    # Track players already scraped to avoid duplicates
    scraped_players = set()
    
    for season in SEASONS:
        print(f"\n=== Season {season} ===")
        
        for team in TEAMS:
            print(f"  Fetching {team}...")
            
            players = get_team_players(team, season)
            
            if not players:
                print(f"  No players found for {team} {season}")
                continue
            
            for player_name in players:
                
                # Skip if already scraped this player
                if player_name in scraped_players:
                    continue
                
                print(f"    Scraping {player_name}...")
                
                games_df, opponent_stats, venue_stats = get_all_player_data(player_name)
                
                if not games_df.empty:
                    all_games.append(games_df)
                
                if opponent_stats:
                    all_opponent_stats[player_name] = opponent_stats
                
                if venue_stats:
                    all_venue_stats[player_name] = venue_stats
                
                scraped_players.add(player_name)
    
    print("\n=== Saving data ===")
    
    # Combine all game dataframes into one big dataframe
    if all_games:
        games_combined = pd.concat(all_games, ignore_index=True)
        games_path = os.path.join(RAW_DATA_PATH, "all_games.csv")
        games_combined.to_csv(games_path, index=False)
        print(f"Saved {len(games_combined)} games to {games_path}")
    
    # Save opponent stats
    opponent_df = pd.DataFrame(all_opponent_stats).T
    opponent_path = os.path.join(RAW_DATA_PATH, "opponent_stats.csv")
    opponent_df.to_csv(opponent_path)
    print(f"Saved opponent stats to {opponent_path}")
    
    # Save venue stats
    venue_df = pd.DataFrame(all_venue_stats).T
    venue_path = os.path.join(RAW_DATA_PATH, "venue_stats.csv")
    venue_df.to_csv(venue_path)
    print(f"Saved venue stats to {venue_path}")
    
    print("\n=== Complete ===")

