import openai
import pandas as pd
import os

from TOKEN import API_KEY

# Set up OpenAI API credentials
openai.api_key = API_KEY

# Define function to analyze text and return a rating from 1 to 10
def analyze_review(review):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=(f"Rate the following review on a scale of 1 to 10, with 10 being the most positive and 1 being the most negative:\n\n{review}\n\nRating:"),
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0.5,
    )
    rating = response.choices[0].text.strip()
    if rating.isdigit():
        return int(rating)
    else:
        return None

# Read the CSV file
filename = input("Enter the CSV filename (including extension): ")
data = pd.read_csv(filename)

# Analyze the reviews and add a rating column to the dataframe
data["rate"] = data["review text"].apply(analyze_review)

# Sort the dataframe by rating in descending order
data = data.sort_values(by=["rate"], ascending=False)

# Save the analyzed data to a new CSV file
new_filename = os.path.splitext(filename)[0] + "_analyzed.csv"
data.to_csv(new_filename, index=False)

print(f"Analysis complete. The analyzed data has been saved to {new_filename}.")
