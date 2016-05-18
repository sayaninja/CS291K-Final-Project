import json

BUSINESS_DATA_FILE = "./dataset/yelp_academic_dataset_business.json"
REVIEWS_DATA_FILE = "./dataset/yelp_academic_dataset_review.json"

RESTAURANTS_OUTPUT_FILE = "./dataset/restaurants.json"
REVIEWS_OUTPUT_FILE = "./dataset/reviews.json"

# Read business data
print("Importing data...")
restaurants = []
restaurant_ids = []
business_file = open(BUSINESS_DATA_FILE, "r")
for line in business_file:
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
        del business["categories"]

        # Save restaurant data
        restaurants.append(business)
        restaurant_ids.append(business["business_id"])
        continue

print("Imported " + str(len(restaurants)) + " restaurants")
business_file.close()

# Save restaurants to new JSON file
print
print("Writing to business file...")
output_file = open(RESTAURANTS_OUTPUT_FILE, "w")
for rest in restaurants:
    output_file.write(json.dumps(rest) + "\n")
output_file.close()

print("Restaurants saved in " + RESTAURANTS_OUTPUT_FILE)


# Read review data
print
print("Getting reviews for restaurants...")
review_file = open(REVIEWS_DATA_FILE, "r")
reviews = []
review_count = 0
for line in review_file:
    review = json.loads(line)

    # Save review if it belongs to a restaurant
    if review['business_id'] in restaurant_ids:
        # Delete unnecessary features
        del review["votes"]
        del review["user_id"]
        del review["date"]
        del review["type"]
        reviews.append(review)
        review_count += 1

    # Track progress
    if review_count % 1000 == 0:
        print(str(review_count) + " restaurant reviews found.")

review_file.close()

# Save restaurant reviews to new JSON file
print
print("Writing to file...")
output_file = open(REVIEWS_OUTPUT_FILE, "w")
for review in reviews:
    output_file.write(json.dumps(review) + "\n")
output_file.close()

print("Reviews saved in " + REVIEWS_OUTPUT_FILE)