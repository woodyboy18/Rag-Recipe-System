import re


def extract_constraints(query: str):

    query = query.lower()

    constraints = {
        "meal": None,
        "diet": None,
        "protein": None,
        "time": None,
        "calories": None,
        "ingredients_to_avoid": [],
        "cuisine": None
    }

    # ---------------- Meal Type ----------------

    meals = [
        "breakfast",
        "lunch",
        "dinner",
        "snack",
        "dessert"
    ]

    for meal in meals:
        if meal in query:
            constraints["meal"] = meal
            break

    # ---------------- Diet ----------------

    diets = [
        "vegetarian",
        "vegan",
        "keto",
        "low carb",
        "gluten free"
    ]

    for diet in diets:
        if diet in query:
            constraints["diet"] = diet
            break

    # ---------------- Protein ----------------

    if "high protein" in query:
        constraints["protein"] = "high"

    elif "low protein" in query:
        constraints["protein"] = "low"

    # ---------------- Cooking Time ----------------

    match = re.search(r'(\d+)\s*(min|mins|minute|minutes)', query)

    if match:
        constraints["time"] = int(match.group(1))

    # ---------------- Calories ----------------

    match = re.search(r'(\d+)\s*(cal|kcal|calories)', query)

    if match:
        constraints["calories"] = int(match.group(1))

    # ---------------- Ingredient Avoidance ----------------

    avoid_patterns = [
        r"without ([a-zA-Z ]+)",
        r"no ([a-zA-Z ]+)",
        r"don't have ([a-zA-Z ]+)",
        r"do not have ([a-zA-Z ]+)"
    ]

    for pattern in avoid_patterns:

        match = re.search(pattern, query)

        if match:

            ingredient = match.group(1).strip()

            constraints["ingredients_to_avoid"].append(ingredient)

    # ---------------- Cuisine ----------------

    cuisines = [
        "indian",
        "italian",
        "chinese",
        "mexican",
        "thai",
        "american"
    ]

    for cuisine in cuisines:
        if cuisine in query:
            constraints["cuisine"] = cuisine
            break

    return constraints

def merge_constraints(previous_constraints, current_constraints):
    """
    Merge previous constraints with current constraints.

    Current constraints always override previous ones if specified.
    """

    merged = previous_constraints.copy()

    for key, value in current_constraints.items():

        if value is None:
            continue

        if isinstance(value, list):

            merged.setdefault(key, [])

            for item in value:
                if item not in merged[key]:
                    merged[key].append(item)

        else:

            merged[key] = value

    return merged
