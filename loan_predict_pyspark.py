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
# Using the below IQR method, I found the following outlier percentages:
# - ApplicantIncome: 50 outliers (~8.1%)
# - CoapplicantIncome: 18 outliers (~2.9%)
# - LoanAmount: 41 outliers (~6.7%)
# Since these are relatively small proportions and may reflect real variation (e.g., high earners or 
# larger loans), I chose to keep the outliers. Removing them could harm the model’s ability to 
# generalize to real-world data.

with open("outliers_log.txt", "w") as f:
    for colname in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
        q1, q3 = df.approxQuantile(colname, [0.25, 0.75], 0.0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = df.filter((col(colname) < lower_bound) | (col(colname) > upper_bound)).count()
        f.write(f"{colname}: {outlier_count} outliers out of {total_rows} rows\n")

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df)

# Split into train/test sets
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Evaluation metrics setup
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_prec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
evaluator_rec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

def evaluate(predictions):
    return {
        "accuracy": evaluator_acc.evaluate(predictions),
        "f1": evaluator_f1.evaluate(predictions),
        "precision": evaluator_prec.evaluate(predictions),
        "recall": evaluator_rec.evaluate(predictions)
    }

results = {}

# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_preds = lr.fit(train).transform(test)
results["Logistic Regression"] = evaluate(lr_preds)

# Decision Tree
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
dt_preds = dt.fit(train).transform(test)
results["Decision Tree"] = evaluate(dt_preds)

# Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)
rf_preds = rf.fit(train).transform(test)
results["Random Forest"] = evaluate(rf_preds)

# Save results
with open("output.txt", "w") as f:
    for model, metrics in results.items():
        f.write(f"{model}:\n")
        for metric, val in metrics.items():
            f.write(f"  {metric.capitalize()}: {val:.4f}\n")
        f.write("\n")

# Print results
print("=== Model Performance ===")
for model, metrics in results.items():
    print(f"{model}:")
    for metric, val in metrics.items():
        print(f"  {metric.capitalize()}: {val:.4f}")
    print()

# Stop Spark session
spark.stop()




