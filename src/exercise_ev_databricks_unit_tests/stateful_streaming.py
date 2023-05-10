from typing import Callable
from pyspark.sql.types import StringType, StructField, StructType, LongType
import pandas as pd


def test_read_from_stream(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "value": 0,
            "charge_point_id": "430ca2a2-54a6-4247-9adf-d86300231c62",
            "action": "StatusNotification",
            "message_type": 2,
            "body": "123"
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType
            ([
            StructField("value", LongType(), True),
            StructField("charge_point_id", StringType(), True),
            StructField("action", StringType(), True),
            StructField("message_type", LongType(), True),
            StructField("body", StringType(), True)
        ])
    )

    print("-->Input Schema")
    input_df.printSchema()

    result = input_df.transform(f)
    print("Transformed DF")

    print("-->Result Schema")
    result.printSchema()

    result_schema = result.schema
    expected_schema = StructType(
        [
            StructField("value", LongType(), True),
            StructField("charge_point_id", StringType(), True),
            StructField("action", StringType(), True),
            StructField("message_type", LongType(), True),
            StructField("body", StringType(), True)
        ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")