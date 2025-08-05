import os
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"  # or the output of `which python3`

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, count
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Step 1: Start Spark session
spark = SparkSession.builder.appName("LoanPrediction").getOrCreate()

# Step 2: Load dataset
df = spark.read.csv("loan_data.csv", header=True, inferSchema=True)

# Step 3: Define features and target
features = [
    "Credit_History",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term"
]
label_column = "Loan_Status"
model_columns = features + [label_column]

# Step 4: Show missing value counts and percentages
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

# Step 5: Handle missing values column by column
# Justification: We drop rows with missing values in essential columns with low missingness,
# and impute those with more frequent/middle values.

# Drop rows where target or income columns are missing
df = df.dropna(subset=["Loan_Status", "ApplicantIncome", "CoapplicantIncome"])

# Impute Credit_History with mode (1.0 = most common/good history)
df = df.fillna({"Credit_History": 1.0})

# Impute LoanAmount with median
median_loan = df.approxQuantile("LoanAmount", [0.5], 0.0)[0]
df = df.fillna({"LoanAmount": median_loan})

# Impute Loan_Amount_Term with mode = 360 months (30 years)
df = df.fillna({"Loan_Amount_Term": 360.0})

# Step 6: Encode label
# Justification: Loan_Status is categorical (Y/N), so we encode it into 0/1
label_indexer = StringIndexer(inputCol="Loan_Status", outputCol="label", handleInvalid="keep")
df = label_indexer.fit(df).transform(df)

# Step 7: Assemble features into a single vector
assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df)

# Step 8: Split into train/test sets
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Step 9: Set up evaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
results = {}

# Step 10: Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train)
lr_preds = lr_model.transform(test)
results["Logistic Regression"] = evaluator.evaluate(lr_preds)

# Step 11: Decision Tree
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
dt_model = dt.fit(train)
dt_preds = dt_model.transform(test)
results["Decision Tree"] = evaluator.evaluate(dt_preds)

# Step 12: Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)
rf_model = rf.fit(train)
rf_preds = rf_model.transform(test)
results["Random Forest"] = evaluator.evaluate(rf_preds)

# Step 13: Save accuracy scores to output.txt
with open("output.txt", "w") as f:
    for model_name, accuracy in results.items():
        f.write(f"{model_name} Accuracy: {accuracy:.4f}\n")

# Done
spark.stop()

