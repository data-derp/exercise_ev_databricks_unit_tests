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
        StructField('message_id', StringType(), True),
        StructField('message_type', IntegerType(), True),
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
    expected_count = 10
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_charge_point_id = [x["charge_point_id"] for x in input_df.select("charge_point_id").collect()]
    expected_charge_point_id = [
        '0f6989c0-2bed-45b7-86e1-c956abfb6cff',
        '111713e6-4291-41a8-ac04-d51e2a4f65d7',
        '240f1d92-8383-446f-8887-82ac71732fb9',
        '38e85852-2c43-4c71-bbe5-3f656e25a20c',
        '5366cb82-e350-4f79-ad5d-9e2b446706cc',
        '8eb888ca-e491-4c2e-a8ce-7b91306924cd',
        '90e077f6-0a92-4c92-8c66-96c326935d47',
        'a1a62472-88bc-4495-a639-f2217394f4d0',
        'cf6ba6fb-1f8f-4194-a25b-fd2477f80862',
        'dcb75c05-903c-4ed9-b44a-8b3a96dab4d4'
    ]
    assert result_charge_point_id == expected_charge_point_id, f"Expected {expected_charge_point_id}, but got {result_charge_point_id}"

    result_timestamp = [x.converted_timestamp for x in input_df.select("converted_timestamp").collect()]
    expected_timestamp = [
        datetime(2023, 1, 6, 23, 37, 7, 5),
        datetime(2023, 1, 6, 23, 37, 50, 6),
        datetime(2023, 1, 6, 23, 1, 23, 27),
        datetime(2023, 1, 6, 23, 57, 37, 25),
        datetime(2023, 1, 6, 23, 52, 30, 13),
        datetime(2023, 1, 6, 23, 47, 59, 3),
        datetime(2023, 1, 6, 23, 20, 37, 25),
        datetime(2023, 1, 6, 23, 34, 44, 42),
        datetime(2023, 1, 6, 23, 8, 9, 38),
        datetime(2023, 1, 6, 23, 36, 16, 1)
    ]
    assert result_timestamp == expected_timestamp, f"Expected {expected_timestamp}, but got {result_timestamp}"

    result_columns = input_df.columns
    expected_columns = ["message_id", "message_type", "charge_point_id", "action", "write_timestamp", "body",
                        "converted_timestamp"]
    assert result_columns == expected_columns, f"Expected {expected_columns}, but got {result_columns}"
    print("All tests pass! :)")

