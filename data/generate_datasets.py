"""Generate benchmark CSV datasets for the reproducibility study.

This script writes prompt datasets used by the baseline evaluation:
- addition_data.csv for arithmetic questions (scaled to 1,000 rows)
- units_data.csv for physics-unit questions (scaled to 1,000 rows)
- capitals_data.csv for geography prompts (scaled to 1,000 rows)
"""

import random
import os
import json
import pandas as pd
import argparse

SEED = 787
random.seed(SEED)

US_STATE_CITIES = {
    "Alabama": ["Birmingham", "Mobile", "Huntsville", "Tuscaloosa", "Auburn", "Decatur", "Dothan", "Florence", "Gadsden", "Hoover"],
    "Alaska": ["Anchorage", "Fairbanks", "Badger", "Knik-Fairview", "College", "Sitka", "Ketchikan", "Wasilla", "Kenai", "Kodiak"],
    "Arizona": ["Tucson", "Mesa", "Chandler", "Glendale", "Scottsdale", "Gilbert", "Tempe", "Peoria", "Surprise", "Yuma"],
    "Arkansas": ["Fort Smith", "Fayetteville", "Springdale", "Jonesboro", "Rogers", "Conway", "Bentonville", "Pine Bluff", "Hot Springs"],
    "California": ["Los Angeles", "San Francisco", "San Diego", "San Jose", "Fresno", "Oakland", "Long Beach", "Bakersfield", "Anaheim", "Santa Ana"],
    "Colorado": ["Colorado Springs", "Aurora", "Fort Collins", "Lakewood", "Thornton", "Arvada", "Westminster", "Pueblo", "Centennial", "Boulder"],
    "Connecticut": ["Bridgeport", "New Haven", "Stamford", "Waterbury", "Norwalk", "Danbury", "New Britain", "Meriden", "Bristol", "West Haven"],
    "Delaware": ["Wilmington", "Newark", "Middletown", "Smyrna", "Milford", "Seaford", "Georgetown", "Elsmere", "New Castle", "Millsboro"],
    "Florida": ["Jacksonville", "Miami", "Tampa", "Orlando", "St. Petersburg", "Hialeah", "Port St. Lucie", "Cape Coral", "Pembroke Pines", "Fort Lauderdale"],
    "Georgia": ["Augusta", "Columbus", "Macon", "Savannah", "Athens", "Sandy Springs", "South Fulton", "Roswell", "Johns Creek", "Warner Robins"],
    "Hawaii": ["Hilo", "Kailua", "Kaneohe", "Kahului", "Kihei", "Kapolei", "Mililani Mauka", "Lihue", "Wailuku", "Pearl City"],
    "Idaho": ["Nampa", "Meridian", "Idaho Falls", "Pocatello", "Caldwell", "Coeur d'Alene", "Twin Falls", "Post Falls", "Lewiston", "Eagle"],
    "Illinois": ["Chicago", "Aurora", "Rockford", "Joliet", "Naperville", "Peoria", "Elgin", "Waukegan", "Cicero"],
    "Indiana": ["Fort Wayne", "Evansville", "South Bend", "Carmel", "Fishers", "Bloomington", "Hammond", "Gary", "Lafayette", "Muncie"],
    "Iowa": ["Cedar Rapids", "Davenport", "Sioux City", "Waterloo", "Iowa City", "Council Bluffs", "Ames", "Dubuque", "Ankeny", "West Des Moines"],
    "Kansas": ["Wichita", "Overland Park", "Kansas City", "Olathe", "Lawrence", "Shawnee", "Lenexa", "Manhattan", "Salina", "Hutchinson"],
    "Kentucky": ["Louisville", "Lexington", "Bowling Green", "Owensboro", "Covington", "Hopkinsville", "Richmond", "Florence", "Georgetown", "Elizabethtown"],
    "Louisiana": ["New Orleans", "Shreveport", "Lafayette", "Lake Charles", "Kenner", "Bossier City", "Monroe", "Alexandria", "Houma", "New Iberia"],
    "Maine": ["Portland", "Lewiston", "Bangor", "South Portland", "Auburn", "Biddeford", "Sanford", "Brunswick", "Scarborough", "Saco"],
    "Maryland": ["Baltimore", "Frederick", "Gaithersburg", "Rockville", "Bowie", "Hagerstown", "Annapolis", "Salisbury", "College Park", "Greenbelt"],
    "Massachusetts": ["Worcester", "Springfield", "Lowell", "Cambridge", "New Bedford", "Brockton", "Quincy", "Lynn", "Fall River", "Newton"],
    "Michigan": ["Detroit", "Grand Rapids", "Warren", "Sterling Heights", "Ann Arbor", "Flint", "Dearborn", "Livonia", "Troy"],
    "Minnesota": ["Minneapolis", "Rochester", "Duluth", "Bloomington", "Brooklyn Park", "Plymouth", "Woodbury", "Maple Grove", "St. Cloud", "Eagan"],
    "Mississippi": ["Gulfport", "Biloxi", "Hattiesburg", "Southaven", "Meridian", "Tupelo", "Olive Branch", "Greenville", "Horn Lake", "Clinton"],
    "Missouri": ["Kansas City", "St. Louis", "Springfield", "Independence", "Columbia", "Lee's Summit", "O'Fallon", "St. Joseph", "St. Charles", "Blue Springs"],
    "Montana": ["Billings", "Missoula", "Great Falls", "Bozeman", "Butte", "Helena", "Kalispell", "Havre", "Anaconda", "Miles City"],
    "Nebraska": ["Omaha", "Bellevue", "Grand Island", "Kearney", "Fremont", "Hastings", "North Platte", "Norfolk", "Columbus", "Papillion"],
    "Nevada": ["Las Vegas", "Henderson", "Reno", "North Las Vegas", "Sparks", "Elko", "Fernley", "Mesquite", "Boulder City"],
    "New Hampshire": ["Manchester", "Nashua", "Derry", "Dover", "Rochester", "Salem", "Merrimack", "Londonderry", "Hudson", "Keene"],
    "New Jersey": ["Newark", "Jersey City", "Paterson", "Elizabeth", "Clifton", "Trenton", "Camden", "Passaic", "Bayonne", "East Orange"],
    "New Mexico": ["Albuquerque", "Las Cruces", "Rio Rancho", "Roswell", "Farmington", "South Valley", "Clovis", "Hobbs", "Alamogordo", "Carlsbad"],
    "New York": ["New York City", "Buffalo", "Rochester", "Yonkers", "Syracuse", "Albany", "New Rochelle", "Mount Vernon", "Schenectady", "Utica"],
    "North Carolina": ["Charlotte", "Greensboro", "Durham", "Winston-Salem", "Fayetteville", "Cary", "Wilmington", "High Point", "Concord", "Asheville"],
    "North Dakota": ["Fargo", "Grand Forks", "Minot", "West Fargo", "Williston", "Dickinson", "Mandán", "Jamestown", "Wahpeton", "Devils Lake"],
    "Ohio": ["Cleveland", "Cincinnati", "Toledo", "Akron", "Dayton", "Parma", "Canton", "Youngstown", "Lorain", "Hamilton"],
    "Oklahoma": ["Tulsa", "Norman", "Broken Arrow", "Lawton", "Edmond", "Moore", "Midwest City", "Enid", "Stillwater", "Muskogee"],
    "Oregon": ["Portland", "Eugene", "Gresham", "Hillsboro", "Beaverton", "Bend", "Medford", "Springfield", "Corvallis", "Albany"],
    "Pennsylvania": ["Philadelphia", "Pittsburgh", "Allentown", "Erie", "Reading", "Scranton", "Bethlehem", "Lancaster", "Harrisburg", "Altoona"],
    "Rhode Island": ["Warwick", "Cranston", "Pawtucket", "East Providence", "Woonsocket", "Coventry", "Cumberland", "North Providence", "West Warwick", "Johnston"],
    "South Carolina": ["Charleston", "North Charleston", "Mount Pleasant", "Rock Hill", "Greenville", "Summerville", "Goose Creek", "Hilton Head Island", "Sumter", "Florence"],
    "South Dakota": ["Sioux Falls", "Rapid City", "Aberdeen", "Brookings", "Watertown", "Mitchell", "Yankton", "Pierre", "Huron", "Spearfish"],
    "Tennessee": ["Memphis", "Knoxville", "Chattanooga", "Clarksville", "Murfreesboro", "Franklin", "Jackson", "Johnson City", "Bartlett", "Hendersonville"],
    "Texas": ["Houston", "San Antonio", "Dallas", "Fort Worth", "El Paso", "Arlington", "Corpus Christi", "Plano", "Lubbock", "Laredo"],
    "Utah": ["West Valley City", "Provo", "West Jordan", "Orem", "Sandy", "Ogden", "St. George", "Layton", "Taylorsville", "South Jordan"],
    "Vermont": ["Burlington", "South Burlington", "Rutland", "Barre", "Winooski", "St. Albans", "Essex Junction", "Bennington"],
    "Virginia": ["Virginia Beach", "Norfolk", "Chesapeake", "Arlington", "Newport News", "Alexandria", "Hampton", "Roanoke", "Portsmouth", "Suffolk"],
    "Washington": ["Seattle", "Spokane", "Tacoma", "Vancouver", "Bellevue", "Kent", "Everett", "Renton", "Federal Way", "Yakima"],
    "West Virginia": ["Huntington", "Morgantown", "Parkersburg", "Wheeling", "Weirton", "Martinsburg", "Beckley", "Clarksburg", "Fairmont", "South Charleston"],
    "Wisconsin": ["Milwaukee", "Green Bay", "Kenosha", "Racine", "Appleton", "Waukesha", "Oshkosh", "Eau Claire", "Janesville", "West Allis"],
    "Wyoming": ["Casper", "Laramie", "Gillette", "Rock Springs", "Sheridan", "Green River", "Evanston", "Riverton", "Jackson", "Cody"]
}

def generate_addition() -> pd.DataFrame:
    rows = []
    # Currently we have ones_a in [0..9] and ones_b in [0..9]. That is 100 pairs.
    # To get exactly 1,000 prompts, we generate exactly 10 examples per pair.
    for ones_a in range(10):
        for ones_b in range(10):
            is_carry = (ones_a + ones_b) >= 10
            # To avoid duplicates, we will sample distinct tens digits for a and b.
            # There are 9 choices for each tens digit (1 to 9).
            # We sample 10 pairs of (tens_a, tens_b) with replacement, but using a seed-derived sequence.
            for _ in range(10):
                tens_a = random.randint(1, 9)
                tens_b = random.randint(1, 9)
                a = tens_a * 10 + ones_a
                b = tens_b * 10 + ones_b
                correct_answer = a + b

                if is_carry:
                    distractor_answer = correct_answer - 10
                else:
                    distractor_answer = correct_answer + 10

                rows.append({
                    "Operand1": a,
                    "Operand2": b,
                    "Answer": str(correct_answer),
                    "DistractorAnswer": str(distractor_answer),
                    "OnesDigitPair": f"{ones_a}+{ones_b}",
                    "sentence": f"Question: What is {a} + {b}? Answer:"
                })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


def generate_units() -> pd.DataFrame:
    domain_map = {
        "temperature": {
            "unit": ["kelvin"], "distractor": "joules",
            "objects": [
                "boiling water", "molten lava", "liquid nitrogen", "stars core",
                "frozen tundra", "campfire", "superconductor", "desert noon",
                "molten copper", "antarctic ice", "deep space", "light bulb filament",
                "human body", "melting glacier", "volcanic vent", "liquid helium",
                "fusion reactor", "cooking oven", "forest fire", "comet tail"
            ]
        },
        "mass": {
            "unit": ["kilograms", "kilogram", "kg"], "distractor": "seconds",
            "objects": [
                "boulder", "anvil", "gold bar", "brick",
                "backpack", "shipping container", "dumbbell", "anchor",
                "paperclip", "feather", "truck", "elephant",
                "grain of sand", "marble", "laptop", "bag of cement",
                "bowling ball", "aircraft carrier", "piano", "suitcase"
            ]
        },
        "time": {
            "unit": ["seconds", "second"], "distractor": "meters",
            "objects": [
                "stopwatch", "camera shutter", "heartbeat", "pendulum swing",
                "reaction time", "countdown", "lightning flash", "sprinter split",
                "orbit period", "laser pulse", "radioactive decay half-life", "sound wave cycle",
                "hourglass run", "camera exposure", "processor cycle", "blink of an eye",
                "match burn", "echo delay", "drum beat", "shutter speed"
            ]
        },
        "force": {
            "unit": ["newtons", "newton"], "distractor": "volts",
            "objects": [
                "rocket engine", "hydraulic press", "magnets pull", "car crash",
                "spring stretch", "hammer strike", "bungee cord", "weightlifter push",
                "gravity pull", "engine thrust", "elastic band recoil", "magnetic repulsion",
                "bullet impact", "water jet", "wind gust", "tug of war",
                "earthquake tremor", "bow string tension", "crane lift", "friction drag"
            ]
        },
        "energy": {
            "unit": ["joules", "joule"], "distractor": "ohms",
            "objects": [
                "battery", "laser beam", "solar panel", "lightning bolt",
                "burning match", "kinetic impact", "exploding firework", "food calorie",
                "battery charge", "photon emission", "chemical reaction", "gasoline combustion",
                "flywheel spin", "compressed spring", "steam turbine", "nuclear fission",
                "falling water", "wind turbine", "heat pump", "coal burning"
            ]
        }
    }

    templates = [
        lambda q, o: f"Fact: The standard scientific unit used to measure the {q} of a {o} is named \"",
        lambda q, o: f"Fact: In physics, when calculating the exact {q} exhibited by a {o}, the result is expressed in the unit named \"",
        lambda q, o: f"Fact: The official SI unit for the {q} of a moving {o} is named \"",
        lambda q, o: f"Fact: To properly quantify the baseline {q} belonging to a {o}, scientists use the unit named \"",
        lambda q, o: f"Fact: The textbook notes that the fundamental unit of measurement for a {o}'s {q} is named \"",
        lambda q, o: f"Fact: When recording the typical {q} associated with a {o}, researchers write it in the unit named \"",
        lambda q, o: f"Fact: The standard metric system unit for checking the {q} of a {o} is called the \"",
        lambda q, o: f"Fact: If a student needs to write down the {q} of a {o}, they should use the unit named \"",
        lambda q, o: f"Fact: In experiments measuring the {q} of a given {o}, the standard unit of choice is \"",
        lambda q, o: f"Fact: A physicist would state that the unit representing the {q} of a {o} is \""
    ]

    rows = []
    # 5 quantities, each with 20 objects, and 10 templates.
    # Total combinations = 5 * 20 * 10 = 1000 prompts.
    for quantity, data in domain_map.items():
        unit = data["unit"]
        distractor = data["distractor"]
        
        for obj in data["objects"]:
            for template_func in templates:
                sentence = template_func(quantity, obj)
                rows.append({
                    "Quantity": quantity,
                    "ContextObject": obj,
                    "Answer": unit,
                    "DistractorAnswer": distractor,
                    "sentence": sentence
                })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


def generate_capitals() -> pd.DataFrame:
    base_file_path = os.path.join(os.path.dirname(__file__), "capitals_base.csv")
    cities_json_path = os.path.join(os.path.dirname(__file__), "cities.json")
    
    if not os.path.exists(base_file_path):
        raise FileNotFoundError(f"Could not find baseline file: {base_file_path}")
    if not os.path.exists(cities_json_path):
        raise FileNotFoundError(f"Could not find cities database file: {cities_json_path}")
        
    df_raw = pd.read_csv(base_file_path)
    
    with open(cities_json_path, "r", encoding="utf-8") as fh:
        cities_data = json.load(fh)
        
    # Build a country to cities lookup map (keys in lowercase)
    country_cities = {}
    for item in cities_data:
        cname = item.get("name", "")
        clist = item.get("cities", [])
        if cname:
            country_cities[cname.lower()] = clist

    # Start with the original rows as the base (189 prompts)
    expanded_rows = []
    for row in df_raw.itertuples(index=False):
        expanded_rows.append({
            "Location": row.Location,
            "Type": row.Type,
            "Answer": row.Answer,
            "DistractorAnswer": row.OtherCity,
            "sentence": f"Fact: The capital of the {row.Type} containing {row.OtherCity} is named"
        })

    # Compile a pool of candidate extra prompts
    candidate_pool = []
    for row in df_raw.itertuples(index=False):
        loc = row.Location
        ltype = row.Type
        ans = row.Answer
        orig_other = row.OtherCity
        
        # Get alternative cities
        alt_cities = []
        if ltype == "state":
            alt_cities = US_STATE_CITIES.get(loc, [])
        elif ltype == "country":
            alt_cities = country_cities.get(loc.lower(), [])
            
        for city in alt_cities:
            # Skip if it is the capital or the original other city
            if city.lower() == ans.lower() or city.lower() == orig_other.lower():
                continue
            # Also filter out extremely long or weirdly formatted city names (e.g. over 30 chars)
            if len(city) > 30:
                continue
            candidate_pool.append((loc, ltype, ans, city))

    # Shuffle candidates to ensure random sampling across states/countries
    random.shuffle(candidate_pool)
    
    # We need exactly 1000 - 189 = 811 extra prompts
    target_extra = 1000 - len(expanded_rows)
    selected_extra = candidate_pool[:target_extra]
    
    for loc, ltype, ans, city in selected_extra:
        expanded_rows.append({
            "Location": loc,
            "Type": ltype,
            "Answer": ans,
            "DistractorAnswer": city,
            "sentence": f"Fact: The capital of the {ltype} containing {city} is named"
        })
        
    df_final = pd.DataFrame(expanded_rows)
    df_final = df_final.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df_final


def main():
    print("Generating Datasets...\n")

    parser = argparse.ArgumentParser(description="Standalone dataset generation script for SAE circuits.")
    parser.add_argument(
        "--capitals", 
        action="store_true", 
        help="Forces processing and extraction of the capital data file alongside default math/unit files."
    )
    args = parser.parse_args()
    output_dir = os.path.dirname(__file__)
    
    if args.capitals:
        print("Flag '--capitals' captured. Building structural sentences for geography data...")
        df_capitals = generate_capitals()
        df_capitals.to_csv(os.path.join(output_dir, "capitals_data.csv"), index=False)
        print(f"Saved {len(df_capitals)} capital entries with text templates.")
    else:
        print("Skipping capital sentence generation. (Pass --capitals flag if needed).")

    df_addition = generate_addition()
    df_addition.to_csv(os.path.join(output_dir, "addition_data.csv"), index=False)
    df_units = generate_units()
    df_units.to_csv(os.path.join(output_dir, "units_data.csv"), index=False)
    print(f"Standalone Mode complete:\nSaved {len(df_addition)} addition problems and \nSaved {len(df_units)} unit problems.")


if __name__ == "__main__":
    main()
