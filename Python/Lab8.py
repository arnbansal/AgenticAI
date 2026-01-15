import pandas as pa

data ={
    "Name":["Arun","ABC","Sakshi", "Sahil"],
    "Age" : [16,14,14,45],
    "Salary": [1000, 2000, 3000, 5000]
}


df = pa.DataFrame(data)

print(df)

print("\n",df["Name"])
print("\n",df.Age)


print("\nFirst row:\n", df.iloc[0])
print("Last two rows:\n", df.tail(2))

print("\nAverage Age:", df["Age"].mean())
print("Maximum Salary:", df["Salary"].max())


print("\nPeople with Salary > 60000:\n", df[df["Salary"] > 3000])


df["Bonus"] = df["Salary"] * 0.1
print("\nDataFrame with Bonus:\n", df)