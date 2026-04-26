import pandas as pd
import ast

# load raw dataset
df = pd.read_csv("data/recipes.csv")

# keep only useful columns (based on your dataset)
df = df[['Name', 'RecipeIngredientParts', 'RecipeInstructions']]
df.dropna(inplace=True)

# sample 35k recipes
df = df.sample(n=35000, random_state=42)

# convert list-like strings to text
def parse_list(x):
    try:
        return ", ".join(ast.literal_eval(x))
    except:
        return str(x)

df['RecipeIngredientParts'] = df['RecipeIngredientParts'].apply(parse_list)
df['RecipeInstructions'] = df['RecipeInstructions'].apply(parse_list)

# create unified document field
df['document'] = (
    "Recipe: " + df['Name'] +
    "\nIngredients: " + df['RecipeIngredientParts'] +
    "\nInstructions: " + df['RecipeInstructions']
)

# rename columns to clean names
df = df.rename(columns={
    'Name': 'title',
    'RecipeIngredientParts': 'ingredients',
    'RecipeInstructions': 'instructions'
})

# save cleaned dataset
df.to_csv("data/recipes_35k_clean.csv", index=False)

print("Saved:", len(df), "recipes")
