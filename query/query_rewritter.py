import re

from substitution.substitutions import SUBSTITUTIONS

def rewrite_query(query, conversation_history, constraints):
    """
    Rewrite conversational queries into retrieval-friendly queries.
    """

    query = query.lower().strip()
    history_queries = []

    for line in conversation_history.split("\n"):

        if line.startswith("User:"):
            history_queries.append(line.lower())

    history = " ".join(history_queries)

    retrieval_parts = []

    # ---------- Recover previous intent ----------

    if "high protein" in history:
        retrieval_parts.append("high protein")

    if "breakfast" in history:
        retrieval_parts.append("breakfast")

    elif "lunch" in history:
        retrieval_parts.append("lunch")

    elif "dinner" in history:
        retrieval_parts.append("dinner")

    elif "snack" in history:
        retrieval_parts.append("snack")

    if "vegetarian" in history:
        retrieval_parts.append("vegetarian")

    if "vegan" in history:
        retrieval_parts.append("vegan")

    # ---------- Current constraints override ----------

    if constraints.get("protein") == "high":
        if "high protein" not in retrieval_parts:
            retrieval_parts.append("high protein")

    if constraints.get("meal"):
        if constraints["meal"] not in retrieval_parts:
            retrieval_parts.append(constraints["meal"])

    if constraints.get("diet"):
        if constraints["diet"] not in retrieval_parts:
            retrieval_parts.append(constraints["diet"])

    if constraints.get("cuisine"):
        retrieval_parts.append(constraints["cuisine"])

    if constraints.get("time"):
        retrieval_parts.append(f"{constraints['time']} minute")

    # ---------- Ingredient avoidance ----------

    patterns = [
        r"don't have (.+)",
        r"do not have (.+)",
        r"without (.+)",
        r"no (.+)"
    ]

    for pattern in patterns:

        match = re.search(pattern, query)

    if match:

        ingredient = match.group(1).strip().lower()

        # Clean extra words that may be captured
        ingredient = ingredient.replace("please generate a recipe", "").strip()

        # Look for a substitution
        substitute = SUBSTITUTIONS.get(ingredient)

        if substitute:
            retrieval_parts.append(substitute)
        else:
            # If no substitute exists, keep the ingredient
            retrieval_parts.append(ingredient)

        return " ".join(dict.fromkeys(retrieval_parts))


    # ---------- Fresh query ----------

    retrieval_parts.append(query)

    return " ".join(dict.fromkeys(retrieval_parts))
