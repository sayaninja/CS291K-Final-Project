import json
import sys

if len(sys.argv) != 2:
    print 'Incorrect counter of arguments.'
    exit()

reviews_path = str(sys.argv[1])

business_ids = ['5UmKMjUEUNdYWqANhGckJw', 'UsFtqoBl7naz8AVUBZMjQQ']

def get_reviews(business_ids):
    reviews = []
    with open(reviews_path) as f:
        for line in f:
            review = json.loads(line)
            if(review['business_id'] in business_ids):
                del review["votes"]
                del review["user_id"]
                del review["date"]
                del review["type"]
                reviews.append(review)
    # print reviews[0]
    # print reviews[1]

get_reviews(business_ids)