from typing import Callable
from pyspark.sql.types import StringType, StructField, StructType, LongType, IntegerType, TimestampType
import pandas as pd
import json
import datetime
from pyspark.sql.functions import from_json, to_timestamp


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

def test_select_columns_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": "430ca2a2-54a6-4247-9adf-d86300231c62",
            "new_body": json.dumps({"status": "SuspendedEV", "timestamp":  "2023-01-01T09:00:00.000+0000"})
        },
         {
            "charge_point_id": "f3cf5e5a-701e-410a-9739-b00cda3f082c",
            "new_body": json.dumps({"status": "Faulted", "timestamp":  "2023-01-01T09:00:00.000+0000"})
        }
    ])

    body_schema = StructType([
            StructField("status", StringType(),True),
            StructField("timestamp", StringType(),True)

        ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("new_body", StringType())
        ])
    ).withColumn("new_body", from_json("new_body", body_schema))

    result = input_df.transform(f)
    print("Transformed DF")
    result.show(truncate=False)

    #Test # 1
    result_count = result.count()
    expected_count = 2
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    #Test # 2
    result_data = [(x.charge_point_id, x.status, x.timestamp) for x in result.collect()]
    expected_data = [("430ca2a2-54a6-4247-9adf-d86300231c62", "SuspendedEV", datetime.datetime(2023, 1, 1, 9, 0)), ("f3cf5e5a-701e-410a-9739-b00cda3f082c", "Faulted", datetime.datetime(2023, 1, 1, 9, 0))]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    #Test # 3
    result_schema = result.schema
    expected_schema = StructType(
        [
            StructField("charge_point_id", StringType(),True),
            StructField("status", StringType(),True),
            StructField("timestamp", TimestampType(),True)
        ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"
    
    print("All tests pass! :)")

def test_aggregate_window_watermark_unit(spark, f: Callable):
    
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": "444984d5-0b9c-474e-a972-71347941ae0e",
            "status": "Reserved",
            "timestamp": "2023-01-01T09:25:00.000+0000",
        },
        {
            "charge_point_id": "444984d5-0b9c-474e-a972-71347941ae0e",
            "status": "Reserved",
            "timestamp": "2022-01-01T09:25:00.000+0000",
        },   
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("status", StringType()),
            StructField("timestamp", StringType())
        ])
    ).withColumn("timestamp", to_timestamp("timestamp"))

    print("-->Input Schema")
    input_df.printSchema()

    result = input_df.transform(f)
    # result = (input_df.transform(f).output("update")) #I have also seen this weird parenthesis execution for streaming
    # result = input_df.transform(f).output("update") #Syed can you check this?
    print("Transformed DF")

    print("-->Result Schema")
    result.printSchema()

    # Schema Shape Test 
    result_schema = result.schema
    expected_schema = StructType(
        [
            StructField("charge_point_id", StringType(),True),
            StructField("status", StringType(),True),
            StructField("window", StructType([
                StructField("start", TimestampType(),True),
                StructField("end", TimestampType(),True)
            ]),False),
            StructField("count(status)", LongType(),False),
        ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    # result_records = [(x.charge_point_id, x.status, x.window.start, x.window.end) for x in result.collect()]
    # expected_records = [
    #     ("444984d5-0b9c-474e-a972-71347941ae0e", "Reserved", datetime.datetime(2023, 1, 1, 9, 25), datetime.datetime(2023, 1, 1, 9, 30))
    # ]
    # assert result_records == expected_records, f"Expected {expected_records}, but got {result_records}"

    # result_count = result.count()
    # expected_count = 1
    # assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"


    print("All tests pass! :)")
