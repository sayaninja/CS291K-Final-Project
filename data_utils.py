import json
import sys

BUSINESS_DATA_FILE = "./dataset/yelp_academic_dataset_business.json"
REVIEWS_DATA_FILE = "./dataset/yelp_academic_dataset_review.json"

# Read in items from JSON file into array
print("Importing data...")
restaurants = []
business_ids = []
businessFile = open(BUSINESS_DATA_FILE, "r")
for line in businessFile:
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
        business_ids.append(business["business_id"])
        continue

print("Imported " + str(len(restaurants)) + " restaurants")
print(restaurants[0])

reviewFile = open(REVIEWS_DATA_FILE, "r")
reviews = []
reviewCount = 0
with open(REVIEWS_DATA_FILE) as f:
    for line in f:
        review = json.loads(line)
        if review['business_id'] in business_ids:
            del review["votes"]
            del review["user_id"]
            del review["date"]
            del review["type"]
            reviews.append(review)
            print("Added review for business: " + str(reviewCount))
            reviewCount += 1
