import json

BUSINESS_DATA_FILE = "./dataset/yelp_academic_dataset_business.json"

# Read in items from JSON file into array
print("Importing data...")
restaurants = []
file = open(BUSINESS_DATA_FILE, "r")
for line in file:
    business = json.loads(line)
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

        # Add business to restaurant array
        restaurants.append(business)
        continue

print("Imported " + str(len(restaurants)) + " restaurants")

# Get reviews for each business

exit(0)