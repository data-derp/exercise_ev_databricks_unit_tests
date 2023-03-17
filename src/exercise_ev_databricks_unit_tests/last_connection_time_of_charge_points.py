from typing import Callable

from pyspark.sql.types import TimestampType, StructType, StructField, StringType, IntegerType
from pyspark.sql import DataFrame

import pandas as pd
from dateutil.parser import parse

from pyspark.sql.types import IntegerType
from datetime import datetime


def test_convert_to_timestamp(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "message_id": "a42ac20e-fe56-4583-9a1e-eb53ac7bc296",
            "message_type": 2,
            "charge_point_id": "AL1000",
            "write_timestamp": "2022-10-02T15:30:17.000345+00:00",
            "action": "Heartbeat",
            "body": "{}"
        },
        {
            "message_id": "f257e3dc-e00c-49d5-982a-c6de791c862f",
            "message_type": 2,
            "charge_point_id": "AL1000",
            "write_timestamp": "2022-10-02T15:32:17.000345+00:00",
            "action": "Heartbeat",
            "body": "{}"
        },
        {
            "message_id": "0c59b722-9807-4eb9-80e4-8178fe5466d5",
            "message_type": 2,
            "charge_point_id": "AL1000",
            "write_timestamp": "2022-10-02T15:34:17.000345+00:00",
            "action": "Heartbeat",
            "body": "{}"
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("message_id", StringType()),
            StructField("message_type", IntegerType()),
            StructField("charge_point_id", StringType()),
            StructField("write_timestamp", StringType()),
            StructField("action", StringType()),
            StructField("body", StringType()),
        ]))

    result = input_df.transform(f)

    print("Transformed DF:")
    result.show()

    result_count = result.count()
    expected_count = 3
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_columns = result.columns
    min_expected_column = "converted_timestamp"
    assert min_expected_column in result_columns, f"Expected minimum {min_expected_column} in {result_columns}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('charge_point_id', StringType(), True),
        StructField('write_timestamp', StringType(), True),
        StructField('action', StringType(), True),
        StructField('body', StringType(), True),
        StructField('converted_timestamp', TimestampType(), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")


def test_most_recent_message_of_charge_point(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": "AL1000",
            "write_timestamp": "2022-10-02T15:30:17.000345+00:00",
            "action": "Heartbeat",
            "body": "{}",
            "converted_timestamp": parse("2022-10-02T15:30:17.000345+00:00")
        },
        {
            "charge_point_id": "AL1000",
            "write_timestamp": "2022-10-02T15:32:17.000345+00:00",
            "action": "Heartbeat",
            "body": "{}",
            "converted_timestamp": parse("2022-10-02T15:32:17.000345+00:00")
        },
        {
            "charge_point_id": "AL2000",
            "write_timestamp": "2022-10-02T15:34:17.000345+00:00",
            "action": "Heartbeat",
            "body": "{}",
            "converted_timestamp": parse("2022-10-02T15:34:17.000345+00:00"),
        },
        {
            "charge_point_id": "AL2000",
            "write_timestamp": "2022-10-02T15:36:17.000345+00:00",
            "action": "Heartbeat",
            "body": "{}",
            "converted_timestamp": parse("2022-10-02T15:36:17.000345+00:00"),
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("write_timestamp", StringType()),
            StructField("action", StringType()),
            StructField("body", StringType()),
            StructField("converted_timestamp", TimestampType()),
        ]))

    result = input_df.transform(f)
    print("Transformed Dataframe:")
    result.show()

    result_count = result.count()
    expected_count = 2
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_columns = result.columns
    expected_columns =  ["charge_point_id", "write_timestamp", "action", "body", "converted_timestamp", "rn"]
    assert result_columns == expected_columns, f"Expected {expected_columns}, but got {result_columns}"
    print("All tests pass! :)")


def test_cleanup(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": "AL1000",
            "write_timestamp": "2022-10-02T15:32:17.000345+00:00",
            "action": "Heartbeat",
            "body": "{}",
            "converted_timestamp": parse("2022-10-02T15:32:17.000345+00:00"),
            "rn": 1
        },
        {
            "charge_point_id": "AL2000",
            "write_timestamp": "2022-10-02T15:36:17.000345+00:00",
            "action": "Heartbeat",
            "body": "{}",
            "converted_timestamp": parse("2022-10-02T15:36:17.000345+00:00"),
            "rn": 1
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("write_timestamp", StringType()),
            StructField("action", StringType()),
            StructField("body", StringType()),
            StructField("converted_timestamp", TimestampType()),
            StructField("rn", IntegerType())
        ]))

    result = input_df.transform(f)

    result_count = result.count()
    expected_count = 2
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_columns = result.columns
    expected_columns = ["charge_point_id", "write_timestamp", "action", "body", "converted_timestamp"]
    assert result_columns == expected_columns, f"Expected {expected_columns}, but got {result_columns}"

    print("All tests pass! :)")


def test_final(input_df: DataFrame):
    result_count = input_df.count()
    expected_count = 5
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_charge_point_id = [x["charge_point_id"] for x in input_df.select("charge_point_id").collect()]
    expected_charge_point_id = ['AL1000', 'AL2000', 'AL3000', 'AL4000', 'AL5000']
    assert result_charge_point_id == expected_charge_point_id, f"Expected {expected_charge_point_id}, but got {result_charge_point_id}"

    result_timestamp = [x.converted_timestamp for x in input_df.select("converted_timestamp").collect()]
    expected_timestamp = [
        datetime(2022, 10, 3, 2, 36, 23, 337),
        datetime(2022, 10, 3, 3, 13, 35, 254),
        datetime(2022, 12, 11, 8, 7, 47, 236),
        datetime(2022, 12, 5, 8, 43, 55, 432),
        datetime(2023, 1, 4, 17, 40, 24, 643631)
    ]
    assert result_timestamp == expected_timestamp, f"Expected {expected_timestamp}, but got {result_timestamp}"

    result_columns = input_df.columns
    expected_columns = ["charge_point_id", "write_timestamp", "action", "body", "converted_timestamp"]
    assert result_columns == expected_columns, f"Expected {expected_columns}, but got {result_columns}"
    print("All tests pass! :)")
