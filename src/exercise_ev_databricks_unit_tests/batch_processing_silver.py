import json
from datetime import datetime
from typing import Callable
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, TimestampType, DoubleType


def test_start_transaction_request_filter_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "action": "StartTransaction",
            "message_type": 2,
            "charge_point_id": "123"
        },
        {
            "action": "StartTransaction",
            "message_type": 3,
            "charge_point_id": "123"
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("action", StringType()),
            StructField("message_type", IntegerType()),
            StructField("charge_point_id", StringType()),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_data = [(x.action, x.message_type) for x in result.collect()]
    expected_data = [("StartTransaction", 2)]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_start_transaction_request_filter_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = input_df

    print("Transformed DF")
    display_f(result)

    result_count = result.count()
    expected_count = 2599
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_action = [x.action for x in result.select("action").distinct().collect()]
    expected_action = ["StartTransaction"]
    assert result_action == expected_action, f"Expected {expected_action}, but got {result_action}"

    result_message_type = [x.message_type for x in result.select("message_type").distinct().collect()]
    expected_message_type = [2]
    assert result_message_type == expected_message_type, f"Expected {expected_message_type}, but got {result_message_type}"

    print("All tests pass! :)")


def test_start_transaction_request_unpack_json_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "action": "StartTransaction",
            "message_type": 2,
            "charge_point_id": "123",
            "body": json.dumps({
                "connector_id": 1,
                "id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
                "meter_start": 0,
                "timestamp": "2022-01-01T08:00:00+00:00",
                "reservation_id": None
            })
        },
        {
            "action": "StartTransaction",
            "message_type": 2,
            "charge_point_id": "123",
            "body": json.dumps({
                "connector_id": 1,
                "id_tag": "7f72c19c-5b36-400e-980d-7a16d30ca490",
                "meter_start": 0,
                "timestamp": "2022-01-01T09:00:00+00:00",
                "reservation_id": None
            })
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("action", StringType()),
            StructField("message_type", IntegerType()),
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

    result_schema = result.schema
    expected_schema = StructType(
        [
            StructField('action', StringType(), True),
            StructField('message_type', IntegerType(), True),
            StructField('charge_point_id', StringType(), True),
            StructField('body', StringType(), True),
            StructField('new_body', StructType([
                StructField('connector_id', IntegerType(), True),
                StructField('id_tag', StringType(), True),
                StructField('meter_start', IntegerType(), True),
                StructField('timestamp', StringType(), True),
                StructField('reservation_id', IntegerType(), True)
            ]), True)
        ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.new_body.timestamp for x in result.collect()]
    expected_data = ['2022-01-01T08:00:00+00:00', '2022-01-01T09:00:00+00:00']
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_start_transaction_request_unpack_json_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = input_df

    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 2599
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('message_id', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('action', StringType(), True),
        StructField('write_timestamp', StringType(), True),
        StructField('body', StringType(), True),
        StructField('new_body', StructType([
            StructField('connector_id', IntegerType(), True),
            StructField('id_tag', StringType(), True),
            StructField('meter_start', IntegerType(), True),
            StructField('timestamp', StringType(), True),
            StructField('reservation_id', IntegerType(), True)
        ]), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.new_body.timestamp for x in result.sort(col("new_body.timestamp")).limit(3).collect()]
    expected_data = ['2023-01-01T10:43:09.900215+00:00', '2023-01-01T11:20:31.296429+00:00',
                     '2023-01-01T14:03:42.294160+00:00']
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_start_transaction_request_flatten_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "action": "StartTransaction",
            "message_type": 2,
            "charge_point_id": "123",
            "body": json.dumps({
                "connector_id": 1,
                "id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
                "meter_start": 0,
                "timestamp": "2022-01-01T08:00:00+00:00",
                "reservation_id": None
            })
        },
        {
            "action": "StartTransaction",
            "message_type": 2,
            "charge_point_id": "123",
            "body": json.dumps({
                "connector_id": 1,
                "id_tag": "7f72c19c-5b36-400e-980d-7a16d30ca490",
                "meter_start": 0,
                "timestamp": "2022-01-01T09:00:00+00:00",
                "reservation_id": None
            })
        },
    ])

    body_schema = StructType([
        StructField("connector_id", IntegerType(), True),
        StructField("id_tag", StringType(), True),
        StructField("meter_start", IntegerType(), True),
        StructField("timestamp", StringType(), True),
        StructField("reservation_id", IntegerType(), True),
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("action", StringType()),
            StructField("message_type", IntegerType()),
            StructField("charge_point_id", StringType()),
            StructField("body", StringType()),
        ])
    ).withColumn("new_body", from_json(col("body"), body_schema))

    result = input_df.transform(f)
    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 2
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('action', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('connector_id', IntegerType(), True),
        StructField('id_tag', IntegerType(), True),
        StructField('meter_start', IntegerType(), True),
        StructField('timestamp', StringType(), True),
        StructField('reservation_id', IntegerType(), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.timestamp for x in result.collect()]
    expected_data = ['2022-01-01T08:00:00+00:00', '2022-01-01T09:00:00+00:00']
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_start_transaction_request_flatten_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = input_df

    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 2599
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('message_id', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('action', StringType(), True),
        StructField('write_timestamp', StringType(), True),
        StructField('connector_id', IntegerType(), True),
        StructField('id_tag', IntegerType(), True),
        StructField('meter_start', IntegerType(), True),
        StructField('timestamp', StringType(), True),
        StructField('reservation_id', IntegerType(), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.timestamp for x in result.sort(col("timestamp")).limit(3).collect()]
    expected_data = ['2023-01-01T10:43:09.900215+00:00', '2023-01-01T11:20:31.296429+00:00',
                     '2023-01-01T14:03:42.294160+00:00']
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_start_transaction_response_filter_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "action": "StartTransaction",
            "message_type": 2,
            "charge_point_id": "123"
        },
        {
            "action": "StartTransaction",
            "message_type": 3,
            "charge_point_id": "123"
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("action", StringType()),
            StructField("message_type", IntegerType()),
            StructField("charge_point_id", StringType()),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_data = [(x.action, x.message_type) for x in result.collect()]
    expected_data = [("StartTransaction", 3)]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_start_transaction_response_filter_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = input_df

    print("Transformed DF")
    display_f(result)

    result_count = result.count()
    expected_count = 2599
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_action = [x.action for x in result.select("action").distinct().collect()]
    expected_action = ["StartTransaction"]
    assert result_action == expected_action, f"Expected {expected_action}, but got {result_action}"

    result_message_type = [x.message_type for x in result.select("message_type").distinct().collect()]
    expected_message_type = [3]
    assert result_message_type == expected_message_type, f"Expected {expected_message_type}, but got {result_message_type}"

    print("All tests pass! :)")


def test_start_transaction_response_unpack_json_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": "123",
            "message_id": "456",
            "body": json.dumps({
                "transaction_id": 1,
                "id_tag_info": {
                    "status": "Accepted",
                    "parent_id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
                    "expiry_date": None
                }
            })
        },
    ])

    body_id_tag_info_schema = StructType([
        StructField("status", StringType(), True),
        StructField("parent_id_tag", StringType(), True),
        StructField("expiry_date", StringType(), True),
    ])

    body_schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("id_tag_info", body_id_tag_info_schema, True)
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("message_id", StringType()),
            StructField("body", StringType()),
        ])
    ).withColumn("new_body", from_json(col("body"), body_schema))

    result = input_df.transform(f)

    print("Transformed DF:")
    result.show()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('charge_point_id', StringType(), True),
        StructField('message_id', StringType(), True),
        StructField('body', StringType(), True),
        StructField('new_body', StructType([
            StructField('transaction_id', IntegerType(), True),
            StructField('id_tag_info', StructType([
                StructField('status', StringType(), True),
                StructField('parent_id_tag', StringType(), True),
                StructField('expiry_date', StringType(), True)
            ]), True)
        ]), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.new_body.id_tag_info.parent_id_tag for x in result.collect()]
    expected_data = ['ea068c10-1bfb-4128-ab88-de565bd5f02f']
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_start_transaction_response_unpack_json_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = input_df

    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 2599
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('message_id', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('action', StringType(), True),
        StructField('write_timestamp', StringType(), True),
        StructField('body', StringType(), True),
        StructField('new_body', StructType([
            StructField('transaction_id', IntegerType(), True),
            StructField('id_tag_info', StructType([
                StructField('status', StringType(), True),
                StructField('parent_id_tag', StringType(), True),
                StructField('expiry_date', StringType(), True)
            ]), True)
        ]), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.new_body.id_tag_info.parent_id_tag for x in result.sort(col("message_id")).limit(3).collect()]
    expected_data = ['0c806549-afb1-4cb4-8b36-77f088e0f273', 'b7bc3b31-5b0d-41b5-b0bf-762ac9b785ed',
                     '9495b2ac-d3ef-4330-a098-f1661ab9303e']
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_start_transaction_response_flatten_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "action": "StartTransaction",
            "message_type": 3,
            "charge_point_id": "123",
            "body": json.dumps({
                "transaction_id": 1,
                "id_tag_info": {
                    "status": "Accepted",
                    "parent_id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
                    "expiry_date": None
                }
            })
        },
        {
            "action": "StartTransaction",
            "message_type": 3,
            "charge_point_id": "123",
            "body": json.dumps({
                "transaction_id": 2,
                "id_tag_info": {
                    "status": "Accepted",
                    "parent_id_tag": "74924177-d936-4898-8943-1c1a512d7f4c",
                    "expiry_date": None
                }
            })
        },
    ])

    body_id_tag_info_schema = StructType([
        StructField("status", StringType(), True),
        StructField("parent_id_tag", StringType(), True),
        StructField("expiry_date", StringType(), True),
    ])

    body_schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("id_tag_info", body_id_tag_info_schema, True)
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("action", StringType()),
            StructField("message_type", IntegerType()),
            StructField("charge_point_id", StringType()),
            StructField("body", StringType()),
        ])
    ).withColumn("new_body", from_json(col("body"), body_schema))

    result = input_df.transform(f)
    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 2
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('action', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('transaction_id', IntegerType(), True),
        StructField('id_tag_info_status', StringType(), True),
        StructField('id_tag_info_parent_id_tag', StringType(), True),
        StructField('id_tag_info_expiry_date', StringType(), True)
    ])

    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.transaction_id for x in result.collect()]
    expected_data = [1, 2]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_start_transaction_response_flatten_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = input_df

    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 2599
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('message_id', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('action', StringType(), True),
        StructField('write_timestamp', StringType(), True),
        StructField('transaction_id', IntegerType(), True),
        StructField('id_tag_info_status', StringType(), True),
        StructField('id_tag_info_parent_id_tag', StringType(), True),
        StructField('id_tag_info_expiry_date', StringType(), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.transaction_id for x in result.sort(col("message_id")).limit(3).collect()]
    expected_data = [1652, 318, 1677]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_stop_transaction_request_filter_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "action": "StopTransaction",
            "message_type": 2,
            "charge_point_id": "123"
        },
        {
            "action": "StopTransaction",
            "message_type": 3,
            "charge_point_id": "123"
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("action", StringType()),
            StructField("message_type", IntegerType()),
            StructField("charge_point_id", StringType()),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_data = [(x.action, x.message_type) for x in result.collect()]
    expected_data = [("StopTransaction", 2)]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_stop_transaction_request_filter_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = input_df

    print("Transformed DF")
    display_f(result)

    result_count = result.count()
    expected_count = 2599
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_action = [x.action for x in result.select("action").distinct().collect()]
    expected_action = ["StopTransaction"]
    assert result_action == expected_action, f"Expected {expected_action}, but got {result_action}"

    result_message_type = [x.message_type for x in result.select("message_type").distinct().collect()]
    expected_message_type = [2]
    assert result_message_type == expected_message_type, f"Expected {expected_message_type}, but got {result_message_type}"

    print("All tests pass! :)")


def test_stop_transaction_request_unpack_json_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "action": "StopTransaction",
            "message_type": 2,
            "charge_point_id": "123",
            "body": json.dumps({
                "meter_stop": 2780,
                "timestamp": "2022-01-01T08:20:00+00:00",
                "transaction_id": 1,
                "reason": None,
                "id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
                "transaction_data": None
            }),
        },
        {
            "action": "StartTransaction",
            "message_type": 2,
            "charge_point_id": "123",
            "body": json.dumps({
                "meter_stop": 5000,
                "timestamp": "2022-01-01T09:20:00+00:00",
                "transaction_id": 1,
                "reason": None,
                "id_tag": "25b72fa9-85fd-4a75-acbe-5a15fc7430a8",
                "transaction_data": None
            }),
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("action", StringType()),
            StructField("message_type", IntegerType()),
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

    result_schema = result.schema
    expected_schema = StructType([
        StructField('action', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('body', StringType(), True),
        StructField('new_body', StructType([
            StructField('meter_stop', IntegerType(), True),
            StructField('timestamp', StringType(), True),
            StructField('transaction_id', IntegerType(), True),
            StructField('reason', StringType(), True),
            StructField('id_tag', StringType(), True),
            StructField('transaction_data', ArrayType(StringType(), True), True)
        ]), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.new_body.timestamp for x in result.collect()]
    expected_data = ['2022-01-01T08:20:00+00:00', '2022-01-01T09:20:00+00:00']
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_stop_transaction_request_unpack_json_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = input_df

    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 2599
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('message_id', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('action', StringType(), True),
        StructField('write_timestamp', StringType(), True),
        StructField('body', StringType(), True),
        StructField('new_body', StructType([
            StructField('meter_stop', IntegerType(), True),
            StructField('timestamp', StringType(), True),
            StructField('transaction_id', IntegerType(), True),
            StructField('reason', StringType(), True),
            StructField('id_tag', StringType(), True),
            StructField('transaction_data', ArrayType(StringType(), True), True)
        ]), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.new_body.timestamp for x in result.sort(col("new_body.timestamp")).limit(3).collect()]
    expected_data = ['2023-01-01T17:56:55.669396+00:00', '2023-01-01T18:31:34.833396+00:00',
                     '2023-01-01T19:10:01.568021+00:00']
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_stop_transaction_request_flatten_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "action": "StopTransaction",
            "message_type": 2,
            "charge_point_id": "123",
            "body": json.dumps({
                "meter_stop": 2780,
                "timestamp": "2022-01-01T08:20:00+00:00",
                "transaction_id": 1,
                "reason": None,
                "id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
                "transaction_data": None
            }),
        },
        {
            "action": "StartTransaction",
            "message_type": 2,
            "charge_point_id": "123",
            "body": json.dumps({
                "meter_stop": 5000,
                "timestamp": "2022-01-01T09:20:00+00:00",
                "transaction_id": 1,
                "reason": None,
                "id_tag": "25b72fa9-85fd-4a75-acbe-5a15fc7430a8",
                "transaction_data": None
            }),
        },
    ])

    body_schema = StructType([
        StructField('meter_stop', IntegerType(), True),
        StructField('timestamp', StringType(), True),
        StructField('transaction_id', IntegerType(), True),
        StructField('reason', StringType(), True),
        StructField('id_tag', StringType(), True),
        StructField('transaction_data', ArrayType(StringType(), True), True)
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("action", StringType()),
            StructField("message_type", IntegerType()),
            StructField("charge_point_id", StringType()),
            StructField("body", StringType()),
        ])
    ).withColumn("new_body", from_json(col("body"), body_schema))

    result = input_df.transform(f)
    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 2
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('action', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('meter_stop', IntegerType(), True),
        StructField('timestamp', StringType(), True),
        StructField('transaction_id', IntegerType(), True),
        StructField('reason', StringType(), True),
        StructField('id_tag', StringType(), True),
        StructField('transaction_data', ArrayType(StringType(), True), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.timestamp for x in result.collect()]
    expected_data = ['2022-01-01T08:20:00+00:00', '2022-01-01T09:20:00+00:00']
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_stop_transaction_request_flatten_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = input_df

    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 2599
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('message_id', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('action', StringType(), True),
        StructField('write_timestamp', StringType(), True),
        StructField('meter_stop', IntegerType(), True),
        StructField('timestamp', StringType(), True),
        StructField('transaction_id', IntegerType(), True),
        StructField('reason', StringType(), True),
        StructField('id_tag', StringType(), True),
        StructField('transaction_data', ArrayType(StringType(), True), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.timestamp for x in result.sort(col("timestamp")).limit(3).collect()]
    expected_data = ['2023-01-01T17:56:55.669396+00:00', '2023-01-01T18:31:34.833396+00:00',
                     '2023-01-01T19:10:01.568021+00:00']
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_meter_values_request_filter_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "action": "MeterValues",
            "message_type": 2,
            "charge_point_id": "123"
        },
        {
            "action": "MeterValues",
            "message_type": 3,
            "charge_point_id": "123"
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("action", StringType()),
            StructField("message_type", IntegerType()),
            StructField("charge_point_id", StringType()),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_data = [(x.action, x.message_type) for x in result.collect()]
    expected_data = [("MeterValues", 2)]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_meter_values_request_filter_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = input_df

    print("Transformed DF")
    display_f(result)

    result_count = result.count()
    expected_count = 218471
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_action = [x.action for x in result.select("action").distinct().collect()]
    expected_action = ["MeterValues"]
    assert result_action == expected_action, f"Expected {expected_action}, but got {result_action}"

    result_message_type = [x.message_type for x in result.select("message_type").distinct().collect()]
    expected_message_type = [2]
    assert result_message_type == expected_message_type, f"Expected {expected_message_type}, but got {result_message_type}"

    print("All tests pass! :)")


def test_meter_values_request_unpack_json_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "action": "MeterValues",
            "message_type": 2,
            "charge_point_id": "123",
            "body": '{"connector_id": 1, "meter_value": [{"timestamp": "2022-10-02T15:30:17.000345+00:00", "sampled_value": [{"value": "0.00", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L1-N", "location": "Outlet", "unit": "V"}, {"value": "13.17", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L1", "location": "Outlet", "unit": "A"}, {"value": "3663.49", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L1", "location": "Outlet", "unit": "W"}, {"value": "238.65", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L2-N", "location": "Outlet", "unit": "V"}, {"value": "14.28", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L2", "location": "Outlet", "unit": "A"}, {"value": "3086.46", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L2", "location": "Outlet", "unit": "W"}, {"value": "215.21", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L3-N", "location": "Outlet", "unit": "V"}, {"value": "14.63", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L3", "location": "Outlet", "unit": "A"}, {"value": "4014.47", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L3", "location": "Outlet", "unit": "W"}, {"value": "254.65", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": null, "location": "Outlet", "unit": "Wh"}, {"value": "11.68", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L1-N", "location": "Outlet", "unit": "V"}, {"value": "3340.61", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L1", "location": "Outlet", "unit": "A"}, {"value": "7719.95", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L1", "location": "Outlet", "unit": "W"}, {"value": "0.00", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L2-N", "location": "Outlet", "unit": "V"}, {"value": "3.72", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L2", "location": "Outlet", "unit": "A"}, {"value": "783.17", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L2", "location": "Outlet", "unit": "W"}, {"value": "242.41", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L3-N", "location": "Outlet", "unit": "V"}, {"value": "3.46", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L3", "location": "Outlet", "unit": "A"}, {"value": "931.52", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L3", "location": "Outlet", "unit": "W"}, {"value": "1330", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": null, "location": "Outlet", "unit": "W"},{"value": "7.26", "context": "Sample.Periodic", "format": "Raw", "measurand": "Energy.Active.Import.Register", "phase": null, "location": "Outlet", "unit": "Wh"}]}], "transaction_id": 1}'
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("action", StringType()),
            StructField("message_type", IntegerType()),
            StructField("charge_point_id", StringType()),
            StructField("body", StringType()),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('action', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('body', StringType(), True),
        StructField('new_body', StructType([
            StructField("connector_id", IntegerType(), True),
            StructField("transaction_id", IntegerType(), True),
            StructField("meter_value", ArrayType(StructType([
                StructField("timestamp", StringType(), True),
                StructField("sampled_value", ArrayType(StructType([
                    StructField("value", StringType(), True),
                    StructField("context", StringType(), True),
                    StructField("format", StringType(), True),
                    StructField("measurand", StringType(), True),
                    StructField("phase", StringType(), True),
                    StructField("unit", StringType(), True)
                ]), True), True)
            ]), True), True)
        ]), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.new_body.transaction_id for x in result.collect()]
    expected_data = [1]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_meter_values_request_unpack_json_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = input_df

    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 218471
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('message_id', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('action', StringType(), True),
        StructField('write_timestamp', StringType(), True),
        StructField('body', StringType(), True),
        StructField('new_body', StructType([
            StructField("connector_id", IntegerType(), True),
            StructField("transaction_id", IntegerType(), True),
            StructField("meter_value", ArrayType(StructType([
                StructField("timestamp", StringType(), True),
                StructField("sampled_value", ArrayType(StructType([
                    StructField("value", StringType(), True),
                    StructField("context", StringType(), True),
                    StructField("format", StringType(), True),
                    StructField("measurand", StringType(), True),
                    StructField("phase", StringType(), True),
                    StructField("unit", StringType(), True)
                ]), True), True)
            ]), True), True)
        ]), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.new_body.transaction_id for x in result.sort(col("message_id")).limit(3).collect()]
    expected_data = [2562, 2562, 2562]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_meter_values_request_flatten_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "action": "MeterValues",
            "message_id": "f8635eee-dd37-4c80-97f9-6a4f1ad3a40b",
            "message_type": 2,
            "charge_point_id": "123",
            "write_timestamp": "2022-10-02T15:30:17.000345+00:00",
            "body": '{"connector_id": 1, "meter_value": [{"timestamp": "2022-10-02T15:30:17.000345+00:00", "sampled_value": [{"value": "0.00", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L1-N", "location": "Outlet", "unit": "V"}, {"value": "13.17", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L1", "location": "Outlet", "unit": "A"}, {"value": "3663.49", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L1", "location": "Outlet", "unit": "W"}, {"value": "238.65", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L2-N", "location": "Outlet", "unit": "V"}, {"value": "14.28", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L2", "location": "Outlet", "unit": "A"}, {"value": "3086.46", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L2", "location": "Outlet", "unit": "W"}, {"value": "215.21", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L3-N", "location": "Outlet", "unit": "V"}, {"value": "14.63", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L3", "location": "Outlet", "unit": "A"}, {"value": "4014.47", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L3", "location": "Outlet", "unit": "W"}, {"value": "254.65", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": null, "location": "Outlet", "unit": "Wh"}, {"value": "11.68", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L1-N", "location": "Outlet", "unit": "V"}, {"value": "3340.61", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L1", "location": "Outlet", "unit": "A"}, {"value": "7719.95", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L1", "location": "Outlet", "unit": "W"}, {"value": "0.00", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L2-N", "location": "Outlet", "unit": "V"}, {"value": "3.72", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L2", "location": "Outlet", "unit": "A"}, {"value": "783.17", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L2", "location": "Outlet", "unit": "W"}, {"value": "242.41", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L3-N", "location": "Outlet", "unit": "V"}, {"value": "3.46", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L3", "location": "Outlet", "unit": "A"}, {"value": "931.52", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L3", "location": "Outlet", "unit": "W"}, {"value": "1330", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": null, "location": "Outlet", "unit": "W"},{"value": "7.26", "context": "Sample.Periodic", "format": "Raw", "measurand": "Energy.Active.Import.Register", "phase": null, "location": "Outlet", "unit": "Wh"}]}], "transaction_id": 1}'
        },
    ])

    body_schema = StructType([
        StructField("connector_id", IntegerType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("meter_value", ArrayType(StructType([
            StructField("timestamp", StringType(), True),
            StructField("sampled_value", ArrayType(StructType([
                StructField("value", StringType(), True),
                StructField("context", StringType(), True),
                StructField("format", StringType(), True),
                StructField("measurand", StringType(), True),
                StructField("phase", StringType(), True),
                StructField("unit", StringType(), True)
            ]), True), True)
        ]), True), True)
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("action", StringType()),
            StructField("message_id", StringType()),
            StructField("message_type", IntegerType()),
            StructField("charge_point_id", StringType()),
            StructField("write_timestamp", StringType()),
            StructField("body", StringType()),
        ])
    ).withColumn("new_body", from_json(col("body"), body_schema))

    result = input_df.transform(f)
    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 21
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('message_id', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('action', StringType(), True),
        StructField('write_timestamp', StringType(), True),
        StructField('transaction_id', IntegerType(), True),
        StructField('timestamp', TimestampType(), True),
        StructField('measurand', StringType(), True),
        StructField('phase', StringType(), True),
        StructField('value', DoubleType(), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.timestamp for x in result.limit(2).collect()]
    expected_data = [datetime(2022, 10, 2, 15, 30, 17, 345), datetime(2022, 10, 2, 15, 30, 17, 345)]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_meter_values_request_flatten_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = input_df

    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 2621652
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('message_id', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('action', StringType(), True),
        StructField('write_timestamp', StringType(), True),
        StructField('transaction_id', IntegerType(), True),
        StructField('timestamp', TimestampType(), True),
        StructField('measurand', StringType(), True),
        StructField('phase', StringType(), True),
        StructField('value', DoubleType(), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.timestamp for x in result.sort(col("timestamp")).limit(3).collect()]
    expected_data = [datetime(2023, 1, 1, 10, 43, 15, 900215), datetime(2023, 1, 1, 10, 43, 15, 900215),
                     datetime(2023, 1, 1, 10, 43, 15, 900215)]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_write_start_transaction_request(spark, dbutils, out_dir):
    df = spark.createDataFrame(dbutils.fs.ls(f"{out_dir}/StartTransactionRequest"))
    df.show()
    snappy_parquet_count = df.filter(col("name").endswith(".snappy.parquet")).count()
    assert snappy_parquet_count == 1, f"Expected 1 .snappy.parquet file, but got {snappy_parquet_count}"

    success_count = df.filter(col("name") == "_SUCCESS").count()
    assert success_count == 1, f"Expected 1 _SUCCESS file, but got {success_count}"

    print("All tests pass! :)")


def test_write_start_transaction_response(spark, dbutils, out_dir):
    df = spark.createDataFrame(dbutils.fs.ls(f"{out_dir}/StartTransactionResponse"))
    df.show()
    snappy_parquet_count = df.filter(col("name").endswith(".snappy.parquet")).count()
    assert snappy_parquet_count == 1, f"Expected 1 .snappy.parquet file, but got {snappy_parquet_count}"

    success_count = df.filter(col("name") == "_SUCCESS").count()
    assert success_count == 1, f"Expected 1 _SUCCESS file, but got {success_count}"

    print("All tests pass! :)")


def test_write_stop_transaction_request(spark, dbutils, out_dir):
    df = spark.createDataFrame(dbutils.fs.ls(f"{out_dir}/StopTransactionRequest"))
    df.show()
    snappy_parquet_count = df.filter(col("name").endswith(".snappy.parquet")).count()
    assert snappy_parquet_count == 1, f"Expected 1 .snappy.parquet file, but got {snappy_parquet_count}"

    success_count = df.filter(col("name") == "_SUCCESS").count()
    assert success_count == 1, f"Expected 1 _SUCCESS file, but got {success_count}"

    print("All tests pass! :)")


def test_write_meter_values_request(spark, dbutils, out_dir):
    df = spark.createDataFrame(dbutils.fs.ls(f"{out_dir}/MeterValuesRequest"))
    df.show()
    snappy_parquet_count = df.filter(col("name").endswith(".snappy.parquet")).count()
    assert snappy_parquet_count == 1, f"Expected 1 .snappy.parquet file, but got {snappy_parquet_count}"

    success_count = df.filter(col("name") == "_SUCCESS").count()
    assert success_count == 1, f"Expected 1 _SUCCESS file, but got {success_count}"

    print("All tests pass! :)")