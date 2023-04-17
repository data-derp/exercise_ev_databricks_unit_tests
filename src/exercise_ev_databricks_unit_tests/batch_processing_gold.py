from datetime import datetime
from typing import Callable, Any, List
import pandas as pd
from dateutil import parser

from pandas import DataFrame, Timestamp
from pyspark.sql.functions import col
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, Row, TimestampType, DoubleType


def test_match_start_transaction_requests_with_responses_unit(spark, f: Callable):
    input_start_transaction_response_pandas = pd.DataFrame([
        {
            "charge_point_id": "123",
            "message_id": "456",
            "transaction_id": 1,
            "id_tag_info_status": "Accepted",
            "id_tag_info_parent_id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
            "id_tag_info_expiry_date": None
        }
    ])

    input_start_transaction_response_df = spark.createDataFrame(
        input_start_transaction_response_pandas,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("message_id", StringType()),
            StructField("transaction_id", IntegerType()),
            StructField("id_tag_info_status", StringType(), True),
            StructField("id_tag_info_parent_id_tag", StringType(), True),
            StructField("id_tag_info_expiry_date", StringType(), True),
        ])
    )

    input_start_transaction_request_pandas = pd.DataFrame([
        {
            "charge_point_id": "123",
            "message_id": "456",
            "connector_id": 1,
            "id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
            "meter_start": 0,
            "timestamp": "2022-01-01T08:00:00+00:00",
            "reservation_id": None
        },
    ])

    input_start_transaction_request_df = spark.createDataFrame(
        input_start_transaction_request_pandas,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("message_id", StringType()),
            StructField("connector_id", IntegerType(), True),
            StructField("id_tag", StringType(), True),
            StructField("meter_start", IntegerType(), True),
            StructField("timestamp", StringType(), True),
            StructField("reservation_id", IntegerType(), True),
        ])
    )

    result = input_start_transaction_response_df.transform(f, input_start_transaction_request_df)

    print("Transformed DF:")
    result.show()

    result_count = result.count()
    assert result_count == 1
    result_schema = result.schema
    expected_schema = StructType([
        StructField('charge_point_id', StringType(), True),
        StructField('transaction_id', IntegerType(), True),
        StructField('meter_start', IntegerType(), True),
        StructField('start_timestamp', StringType(), True),
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")

def test_match_start_transaction_requests_with_responses_e2e(input_df, display_f: Callable, **kwargs):
    result = input_df

    print("Transformed DF")
    result.show()

    result_count = result.count()
    expected_count = 2599
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('charge_point_id', StringType(), True),
        StructField('transaction_id', IntegerType(), True),
        StructField('meter_start', IntegerType(), True),
        StructField('start_timestamp', StringType(), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_data = [x.transaction_id for x in result.sort(col("transaction_id")).limit(3).collect()]
    expected_data = [1, 2, 3]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")


def test_join_with_start_transaction_responses_unit(spark, f: Callable):
    input_start_transaction_pandas = pd.DataFrame([
        {
            "charge_point_id": "123",
            "transaction_id": 1,
            "meter_start": 0,
            "start_timestamp":  "2022-01-01T08:00:00+00:00"
        },
    ])

    input_start_transaction_df = spark.createDataFrame(
        input_start_transaction_pandas,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("transaction_id", IntegerType()),
            StructField("meter_start", IntegerType()),
            StructField("start_timestamp", StringType()),
        ])
    )

    input_stop_transaction_request_pandas = pd.DataFrame([
        {
            "foo": "bar",
            "meter_stop": 2780,
            "timestamp": "2022-01-01T08:20:00+00:00",
            "transaction_id": 1,
            "reason": None,
            "id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
            "transaction_data": None
        }
    ])

    input_stop_transaction_request_schema = StructType([
        StructField("foo", StringType(), True),
        StructField("meter_stop", IntegerType(), True),
        StructField("timestamp", StringType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("reason", StringType(), True),
        StructField("id_tag", StringType(), True),
        StructField("transaction_data", StringType(), True),
    ])

    input_stop_transaction_request_df = spark.createDataFrame(
        input_stop_transaction_request_pandas,
        input_stop_transaction_request_schema
    )


    result = input_stop_transaction_request_df.transform(f, input_start_transaction_df)

    print("Transformed DF:")
    result.show()

    result_count = result.count()
    assert result_count == 1

    result_row = result.collect()[0]
    def assert_row_value(row: Row, field: str, value: Any):
        r = getattr(row, field)
        assert getattr(row, field) == value, f"Expected {value} but got {r}"

    assert_row_value(result_row, "charge_point_id", "123")
    assert_row_value(result_row, "transaction_id", 1)
    assert_row_value(result_row, "meter_start", 0)
    assert_row_value(result_row, "meter_stop", 2780)
    assert_row_value(result_row, "start_timestamp", "2022-01-01T08:00:00+00:00")
    assert_row_value(result_row, "stop_timestamp", "2022-01-01T08:20:00+00:00")

    print("All tests pass! :)")

def test_join_with_start_transaction_responses_e2e(input_df: DataFrame, display_f: Callable, **kwargs):
    result = input_df

    print("Transformed DF:")
    display_f(result)

    assert set(result.columns) == {"charge_point_id", "transaction_id", "meter_start", "meter_stop", "start_timestamp", "stop_timestamp"}
    assert result.count() == 2599, f"expected 95, but got {result.count()}"

    result_sub = result.sort(col("transaction_id")).limit(3)
    print("Reordered DF under test:")
    display_f(result_sub)

    def assert_expected_value(column: str, expected_values: List[Any]):
        values = [getattr(x, column) for x in result_sub.select(col(column)).collect()]
        assert values == expected_values, f"expected {expected_values} in column {column}, but got {values}"

    assert_expected_value("charge_point_id", ['94073806-8222-430e-8ca4-fab78b58fb67', 'acea7af6-eb97-4158-8549-2edda4aab255', '7e8404de-845e-4562-9587-720707e87de8'])
    assert_expected_value("transaction_id", [1, 2, 3])
    assert_expected_value("meter_start", [0, 0, 0])
    assert_expected_value("meter_stop", [95306, 78106, 149223])
    assert_expected_value("start_timestamp", ['2023-01-01T10:43:09.900215+00:00', '2023-01-01T11:20:31.296429+00:00', '2023-01-01T14:03:42.294160+00:00'])
    assert_expected_value("stop_timestamp", ['2023-01-01T18:31:34.833396+00:00', '2023-01-01T17:56:55.669396+00:00', '2023-01-01T23:19:26.063351+00:00'])

    print("All tests pass! :)")

def test_calculate_total_time_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": '123',
            "transaction_id": 1,
            "meter_start": 0,
            "meter_stop": 1000,
            "start_timestamp": datetime.fromisoformat("2023-01-01T08:00:00+00:00"),
            "stop_timestamp": datetime.fromisoformat("2023-01-01T09:00:00+00:00"),
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("transaction_id", IntegerType(), True),
            StructField("meter_start", IntegerType(), True),
            StructField("meter_stop", IntegerType(), True),
            StructField("start_timestamp", TimestampType(), True),
            StructField("stop_timestamp", TimestampType(), True),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF:")
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_values = result.toPandas().to_dict(orient="records")
    expected_values = [{
        "charge_point_id": '123',
        "transaction_id": 1,
        "meter_start": 0,
        "meter_stop": 1000,
        "start_timestamp": Timestamp('2023-01-01 08:00:00.000000'),
        "stop_timestamp": Timestamp('2023-01-01 09:00:00.000000'),
        "total_time": 1.0
    }]
    assert result_values == expected_values, f"expected {expected_values}, but got {result_values}"

    result_total_time = [x.total_time for x in result.collect()]
    expected_total_time = [1.0]
    assert result_total_time == expected_total_time, f"expected {expected_total_time}, but got {result_total_time}"

    print("All tests pass! :)")


def test_calculate_total_time_hours_e2e(input_df: DataFrame, display_f: Callable, **kwargs):
    result = input_df
    print("Transformed DF:")
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 2599
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('charge_point_id', StringType(), True),
        StructField('transaction_id', IntegerType(), True),
        StructField('meter_start', IntegerType(), True),
        StructField('meter_stop', IntegerType(), True),
        StructField('start_timestamp', TimestampType(), True),
        StructField('stop_timestamp', TimestampType(), True),
        StructField('total_time', DoubleType(), True)
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    result_total_time = [x.total_time for x in result.sort(col("transaction_id")).limit(3).collect()]
    expected_total_time = [7.81, 6.61, 9.26]

    assert result_total_time == expected_total_time, f"expected {expected_total_time}, but got {result_total_time}"
    print("All tests passed! :)")


def test_calculate_total_energy_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": '123',
            "transaction_id": 1,
            "meter_start": 0,
            "meter_stop": 1000,
            "start_timestamp": Timestamp('2023-01-01 08:00:00.000000'),
            "stop_timestamp": Timestamp('2023-01-01 09:00:00.000000'),
            "total_time": 1.0
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("transaction_id", IntegerType(), True),
            StructField("meter_start", IntegerType(), True),
            StructField("meter_stop", IntegerType(), True),
            StructField("start_timestamp", TimestampType(), True),
            StructField("stop_timestamp", TimestampType(), True),
            StructField("total_time", DoubleType(), True),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF:")
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_values = result.toPandas().to_dict(orient="records")
    expected_values = [{
        "charge_point_id": '123',
        "transaction_id": 1,
        "meter_start": 0,
        "meter_stop": 1000,
        "start_timestamp": Timestamp('2023-01-01 08:00:00.000000'),
        "stop_timestamp": Timestamp('2023-01-01 09:00:00.000000'),
        "total_time": 1.0,
        "total_energy": 1000.0
    }]
    assert result_values == expected_values, f"expected {expected_values}, but got {result_values}"

    result_total_energy = [x.total_energy for x in result.collect()]
    expect_total_energy = [1000.0]
    assert result_total_energy == expect_total_energy, f"expected {expect_total_energy}, but got {result_total_energy}"

    print("All tests pass! :)")

def test_calculate_total_energy_e2e(input_df: DataFrame, display_f: Callable, **kwargs):
    result = input_df

    print("Transformed DF:")
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 2599
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("charge_point_id", StringType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("meter_start", IntegerType(), True),
        StructField("meter_stop", IntegerType(), True),
        StructField("start_timestamp", TimestampType(), True),
        StructField("stop_timestamp", TimestampType(), True),
        StructField("total_time", DoubleType(), True),
        StructField("total_energy", DoubleType(), True),
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"
    result_ordered = result.sort(col("transaction_id"))
    result_total_energy = [x.total_energy for x in result_ordered.limit(3).collect()]
    expected_total_energy = [95306.0, 78106.0, 149223.0]
    assert result_total_energy == expected_total_energy, f"expected {expected_total_energy}, but got {result_total_energy}"
    print("All tests passed! :)")


def test_calculate_total_parking_time_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "transaction_id": 1,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 1.0,
            "timestamp": parser.parse("2023-01-01T09:00:00Z")
        },
        {
            "transaction_id": 1,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 1.0,
            "timestamp": parser.parse("2023-01-01T09:05:00Z")
        },
        {
            "transaction_id": 1,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 1.0,
            "timestamp": parser.parse("2023-01-01T09:10:00Z")
        },
        {
            "transaction_id": 1,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 0.0,
            "timestamp": parser.parse("2023-01-01T09:15:00Z")
        },
        {
            "transaction_id": 1,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 0.0,
            "timestamp": parser.parse("2023-01-01T09:20:00Z")
        },
        {
            "transaction_id": 1,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 0.0,
            "timestamp": parser.parse("2023-01-01T09:25:00Z")
        },
        {
            "transaction_id": 2,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 2.0,
            "timestamp": parser.parse("2023-01-01T09:00:00Z")
        },
        {
            "transaction_id": 2,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 2.0,
            "timestamp": parser.parse("2023-01-01T09:05:00Z")
        },
        {
            "transaction_id": 2,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 2.0,
            "timestamp": parser.parse("2023-01-01T09:10:00Z")
        },
        {
            "transaction_id": 2,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 0.0,
            "timestamp": parser.parse("2023-01-01T09:15:00Z")
        },
        {
            "transaction_id": 2,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 0.0,
            "timestamp": parser.parse("2023-01-01T09:20:00Z")
        },
        {
            "transaction_id": 2,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 0.0,
            "timestamp": parser.parse("2023-01-01T09:25:00Z")
        },
        {
            "transaction_id": 2,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 0.0,
            "timestamp": parser.parse("2023-01-01T09:30:00Z")
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("transaction_id", IntegerType(), True),
            StructField("measurand", StringType(), True),
            StructField("phase", StringType(), True),
            StructField("value", DoubleType(), True),
            StructField("timestamp", TimestampType(), True),
        ])
    )

    print("Transformed DF:")
    result = input_df.transform(f)
    result.show()

    result_count = result.count()
    expected_count = 2
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    def assert_expected_value(df: DataFrame, column_name: str, expected_values: List[Any]):
        r = [getattr(x, column_name) for x in df.collect()]
        assert r == expected_values, f"Expected {expected_values}, but got {r}"

    assert_expected_value(result, "transaction_id", [1, 2])
    assert_expected_value(result, "total_parking_time", [0.17, 0.25])

    result_schema = result.schema
    expected_schema = StructType([
        StructField('transaction_id', IntegerType(), True),
        StructField('total_parking_time', DoubleType(), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")

def test_calculate_total_parking_time_e2e(input_df: DataFrame, display_f: Callable, **kwargs):
    result = input_df

    display_f(result)

    result_count = result.count()
    expected_count = 2583
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('transaction_id', IntegerType(), True),
        StructField('total_parking_time', DoubleType(), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_total_parking_time = [x.total_parking_time for x in result.limit(3).collect()]
    expected_total_parking_time = [2.17, 1.58, 1.25]
    assert result_total_parking_time == expected_total_parking_time, f"Expected {expected_total_parking_time}, but got {result_total_parking_time}"

    print("All tests pass! :)")

def test_join_and_shape_unit(spark, f: Callable):
    join_df_pandas = pd.DataFrame([
        {
            "transaction_id": 1,
            "total_parking_time": 0.1
        },
        {
            "transaction_id": 3,
            "total_parking_time": 0.2
        }
    ])

    join_df = spark.createDataFrame(
        join_df_pandas,
        StructType([
            StructField("transaction_id", IntegerType(), True),
            StructField("total_parking_time", DoubleType(), True),
        ])
    )

    input_df_pandas = pd.DataFrame([
        {
            "charge_point_id": "123",
            "transaction_id": 1,
            "meter_start": 0,
            "meter_stop": 100,
            "start_timestamp": parser.parse("2023-01-01T09:00:00Z"),
            "stop_timestamp": parser.parse("2023-01-01T09:30:00Z"),
            "total_time": 0.5,
            "total_energy": 100,
        },
        {
            "charge_point_id": "123",
            "transaction_id": 2,
            "meter_start": 0,
            "meter_stop": 200,
            "start_timestamp": parser.parse("2023-01-01T09:00:00Z"),
            "stop_timestamp": parser.parse("2023-01-01T09:30:00Z"),
            "total_time": 0.5,
            "total_energy": 200,
        },
        {
            "charge_point_id": "123",
            "transaction_id": 3,
            "meter_start": 0,
            "meter_stop": 300,
            "start_timestamp": parser.parse("2023-01-01T09:00:00Z"),
            "stop_timestamp": parser.parse("2023-01-01T09:30:00Z"),
            "total_time": 0.5,
            "total_energy": 300,
        }
    ])
    input_df = spark.createDataFrame(
        input_df_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("transaction_id", IntegerType(), True),
            StructField("meter_start", IntegerType(), True),
            StructField("meter_stop", IntegerType(), True),
            StructField("start_timestamp", TimestampType(), True),
            StructField("stop_timestamp", TimestampType(), True),
            StructField("total_time", DoubleType(), True),
            StructField("total_energy", DoubleType(), True),
        ])
    )

    result = input_df.transform(f, join_df)
    print("Transformed DF:")
    result.show()

    result_count = result.count()
    expected_count = 3
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("charge_point_id", StringType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("meter_start", IntegerType(), True),
        StructField("meter_stop", IntegerType(), True),
        StructField("start_timestamp", TimestampType(), True),
        StructField("stop_timestamp", TimestampType(), True),
        StructField("total_time", DoubleType(), True),
        StructField("total_energy", DoubleType(), True),
        StructField("total_parking_time", DoubleType(), True),
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_total_parking_time = [x.total_parking_time for x in result.collect()]
    expected_total_parking_time = [0.1, None, 0.2]
    assert result_total_parking_time == expected_total_parking_time, f"Expected {expected_total_parking_time}, but got {result_total_parking_time}"

    print("All tests pass! :)")

def test_join_and_shape_df_e2e(input_df: DataFrame, display_f: Callable, **kwargs):
    result = input_df

    display_f(result)

    result_schema = result.schema
    expected_schema = StructType([
        StructField("charge_point_id", StringType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("meter_start", IntegerType(), True),
        StructField("meter_stop", IntegerType(), True),
        StructField("start_timestamp", TimestampType(), True),
        StructField("stop_timestamp", TimestampType(), True),
        StructField("total_time", DoubleType(), True),
        StructField("total_energy", DoubleType(), True),
        StructField("total_parking_time", DoubleType(), True),
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_total_parking_time = [x.total_parking_time for x in result.limit(3).collect()]
    expected_total_parking_time = [2.17, 1.58, 1.0]
    assert result_total_parking_time == expected_total_parking_time, f"Expected {expected_total_parking_time}, but got {result_total_parking_time}"

    print("All tests pass!")