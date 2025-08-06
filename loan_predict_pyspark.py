from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, count
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Start Spark session
spark = SparkSession.builder.appName("LoanPrediction").getOrCreate()

# Load dataset
df = spark.read.csv("loan_data.csv", header=True, inferSchema=True)

# Drop columns that will not be used. 
# I don't need 'Loan:ID', because it is a primary key for the database, but does not play any role in 
# model building
# I dropped 'Gender', 'Married', 'Education', 'Self_Employed', 'Proeperty_area' because  they are either 
# demographic or indirect and may bias the model unfairly.

df = df.drop("Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area")


# Define features and target
# Credit history shows how loans worked in the past, it is a good predictor of an outcome.
# Total available income (applicant + coapplicant income) shows resources available to applicant
# Loan amount and term shows the load that applicant is undertaking. 

features = [
    "Credit_History",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term"
]
label_column = "Loan_Status"
model_columns = features + [label_column]

# Show missing value counts and percentages
# I wanted to see the quality of my data. After running this script, I saw that I don't have missing data in
# values in income (both applucant and coapplicant) and loan status. 
# I have missing values in credit history and loan amount and term. 
total_rows = df.count()
print(f"Total rows: {total_rows}")

missing_counts = df.select([
    count(when(col(c).isNull(), c)).alias(c + "_missing") for c in model_columns
])
missing_counts.show()

missing_percentages = df.select([
    (count(when(col(c).isNull(), c)) / total_rows).alias(c + "_missing_pct") for c in model_columns
])
missing_percentages.show()

# Handle missing values column by column
# I assume that credit history 0 is bad and credit history 1 is # good. I don't know what happens
# if a person does not have credit history, because it is the first credit. Would it be a missing
# value? I have about 8% of missing credit history, so I replaced missing values with 1 instead of 
# dropping them, assuming that good credit history is more common than bad one.
# Missing values in loan amount and loan term are imputed to avoid losing too many rows. Imputation
# values are based on median in case of loan amount and on mode (360 months) in case of loan term.


# Impute Credit_History with mode (1.0 = most common/good history)
df = df.fillna({"Credit_History": 1.0})

# Impute LoanAmount with median
median_loan = df.approxQuantile("LoanAmount", [0.5], 0.0)[0]
df = df.fillna({"LoanAmount": median_loan})

# Impute Loan_Amount_Term with mode = 360 months (30 years)
df = df.fillna({"Loan_Amount_Term": 360.0})

# Encode label
# Loan_Status is categorical (Y/N), so I encode it into 0/1
label_indexer = StringIndexer(inputCol="Loan_Status", outputCol="label", handleInvalid="keep")
df = label_indexer.fit(df).transform(df)

# I want to check how big is my dataset to make informed decision about the outliers.
total_rows = df.count()
print(f"Total rows in dataset: {total_rows}")

# How big of a problem are outliers?
for colname in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
    q1, q3 = df.approxQuantile(colname, [0.25, 0.75], 0.0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = df.filter((col(colname) < lower_bound) | (col(colname) > upper_bound)).count()
    print(f"{colname}: {outlier_count} outliers out of {total_rows} rows")

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df)

# Split into train/test sets
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Set up evaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
results = {}

# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train)
lr_preds = lr_model.transform(test)
results["Logistic Regression"] = evaluator.evaluate(lr_preds)

# Decision Tree
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
dt_model = dt.fit(train)
dt_preds = dt_model.transform(test)
results["Decision Tree"] = evaluator.evaluate(dt_preds)

# Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)
rf_model = rf.fit(train)
rf_preds = rf_model.transform(test)
results["Random Forest"] = evaluator.evaluate(rf_preds)

# Save accuracy scores to output.txt
with open("output.txt", "w") as f:
    for model_name, accuracy in results.items():
        f.write(f"{model_name} Accuracy: {accuracy:.4f}\n")

# print accuracy scores on the screen after running the script
print("=== Model Accuracy Scores ===")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")

spark.stop()



