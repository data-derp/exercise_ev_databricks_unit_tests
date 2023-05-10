from typing import Callable
from pyspark.sql.types import StringType, StructField, StructType, LongType, IntegerType
import pandas as pd
import json


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


def test_unpack_json_from_status_notification_request_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": "123",
            "body": json.dumps({
                    "connector_id": 1, 
                    "error_code": "NoError", 
                    "status": "Available", 
                    "timestamp": "2023-01-01T09:00:00+00:00", 
                    "info": None, 
                    "vendor_id": None, 
                    "vendor_error_code": None
                })
        },
        {
            "charge_point_id": "456",
            "body": json.dumps({
                "connector_id": 1, 
                "error_code": "NoError", 
                "status": "SuspendedEVSE", 
                "timestamp": "2023-01-01T09:03:00+00:00", 
                "info": None, 
                "vendor_id": None, 
                "vendor_error_code": None
            })
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("body", StringType()),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 2
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_data = [(x.charge_point_id, x.new_body.status) for x in result.collect()]
    expected_data = [("123", "Available"), ("456", "SuspendedEVSE")]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    result_schema = result.schema
    expected_schema = StructType(
        [
            StructField("charge_point_id", StringType(),True),
            StructField("body", StringType(),True),
            StructField("new_body", StructType([
                StructField("connector_id", IntegerType(),True),
                StructField("error_code", StringType(),True),
                StructField("status", StringType(),True),
                StructField("timestamp", StringType(),True),
                StructField("info", IntegerType(),True),
                StructField("vendor_id", IntegerType(),True),
                StructField("vendor_error_code", IntegerType(),True)
            ]),True)
        ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")