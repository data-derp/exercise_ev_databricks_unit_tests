from typing import Callable
import pandas as pd
from datetime import datetime

from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, StructField, StructType


def test_create_partition_columns_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "write_timestamp": "2023-01-01T09:00:00Z",
            "action": "bar",
            "charge_point_id": "123"
        },
        {
            "write_timestamp": "2023-01-01T10:00:00Z",
            "action": "bar",
            "charge_point_id": "123"
        },
        {
            "write_timestamp": "2023-01-01T10:30:00Z",
            "action": "bar",
            "charge_point_id": "123"
        },

    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("write_timestamp", StringType()),
            StructField("action", StringType()),
            StructField("charge_point_id", StringType()),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 3
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_columns = result.columns
    expected_columns = ['write_timestamp', 'action', 'charge_point_id', 'year', 'month', 'day', 'hour']
    assert result_columns == expected_columns, f"expected {expected_columns}, but got {result_columns}"

    result_data = [(x.write_timestamp, x.year, x.month, x.day, x.hour) for x in result.collect()]
    expected_data = [
        (datetime(2023, 1, 1, 9, 0), 2023, 1, 1, 9),
        (datetime(2023, 1, 1, 10, 0), 2023, 1, 1, 10),
        (datetime(2023, 1, 1, 10, 30), 2023, 1, 1, 10)
    ]
    assert result_data == expected_data, f"expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_create_partition_columns_e2e(input_df: DataFrame, display_f, **kwargs):
    result = input_df
    print("Transformed DF:")
    display_f(result)

    result_columns = result.columns
    expected_columns = ['message_id', 'message_type', 'charge_point_id', 'action', 'write_timestamp', 'body', 'year',
                        'month', 'day', 'hour']
    assert result_columns == expected_columns, f"expected {expected_columns}, but got {result_columns}"

    result_sub = result.limit(3)
    result_data = [(x.write_timestamp, x.year, x.month, x.day, x.hour) for x in result_sub.collect()]
    expected_data = [
        (datetime(2023, 1, 1, 8, 3, 45), 2023, 1, 1, 8),
        (datetime(2023, 1, 1, 8, 3, 46), 2023, 1, 1, 8),
        (datetime(2023, 1, 1, 8, 4, 45), 2023, 1, 1, 8)]

    assert result_data == expected_data, f"expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_write_yyyy_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = spark.createDataFrame(input_df)
    display_f(result)
    result_count = result.count()
    expected_count = 6
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_name = [x.name for x in result.collect()]
    expected_name = ['month=1/', 'month=2/', 'month=3/', 'month=4/', 'month=5/', 'month=6/']
    assert result_name == expected_name, f"Expected {expected_name}, but got {result_name}"

    print("All tests pass! :)")


def test_write_mm_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = spark.createDataFrame(input_df)
    display_f(result)
    result_count = result.count()
    expected_count = 31
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_name = [x.name for x in result.collect()]
    expected_name = ['day=1/', 'day=10/', 'day=11/', 'day=12/', 'day=13/', 'day=14/', 'day=15/', 'day=16/', 'day=17/',
                     'day=18/', 'day=19/', 'day=2/', 'day=20/', 'day=21/', 'day=22/', 'day=23/', 'day=24/', 'day=25/',
                     'day=26/', 'day=27/', 'day=28/', 'day=29/', 'day=3/', 'day=30/', 'day=31/', 'day=4/', 'day=5/',
                     'day=6/', 'day=7/', 'day=8/', 'day=9/']
    assert result_name == expected_name, f"Expected {expected_name}, but got {result_name}"

    print("All tests pass! :)")


def test_write_dd_e2e(input_df: DataFrame, spark, display_f, **kwargs):
    result = spark.createDataFrame(input_df)
    display_f(result)
    result_count = result.count()
    expected_count = 16
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_name = [x.name for x in result.collect()]
    expected_name = ['hour=10/', 'hour=11/', 'hour=12/', 'hour=13/', 'hour=14/', 'hour=15/', 'hour=16/', 'hour=17/',
                     'hour=18/', 'hour=19/', 'hour=20/', 'hour=21/', 'hour=22/', 'hour=23/', 'hour=8/', 'hour=9/']
    assert result_name == expected_name, f"Expected {expected_name}, but got {result_name}"

    print("All tests pass! :)")