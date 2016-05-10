import json
import numpy as np

BUSINESS_DATA_FILE = "./dataset/yelp_academic_dataset_business.json"

# Read in JSON file into array
print("Importing data...")
businesses = []
file = open(BUSINESS_DATA_FILE, "r")
for line in file:
    businesses.append(json.loads(line))
print("Imported " + str(len(businesses)) + " businesses")
print(businesses[0]["categories"])

# Subset the data. Remove non restaurant businesses.
print
print("Removing non-restaurants...")
restaurants = []
for business in businesses:
    # Delete if not a restaurant
    if ('Restaurants' in business["categories"]) and \
                    business["review_count"] > 0:
        # Delete unnecessary attributes to save space
        del business["type"]
        del business["name"]
        del business["neighborhoods"]
        del business["full_address"]
        del business["city"]
        del business["state"]
        del business["longitude"]
        del business["latitude"]
        del business["open"]
        del business["hours"]
        del business["attributes"]
        restaurants.append(business)
        continue
print(str(len(restaurants)) + " businesses are restaurants")
del businesses # Free some memory

# Get reviews for each business

exit(0)