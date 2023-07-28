from typing import Callable, Any, List
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType, TimestampType
import pandas as pd
import json
from pyspark.sql.functions import col, from_json, to_timestamp
from pyspark.sql import DataFrame
from datetime import datetime
from pandas import Timestamp
from pyspark.sql import Row
from dateutil import parser


def test_return_stop_transaction_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "foo": "30e2ed0c-dd61-4fc1-bcb8-f0a8a0f87c0a",
            "action": "bar",
        },
        {
            "foo": "4496309f-dfc5-403d-a1c1-54d21b9093c1",
            "action": "StopTransaction",
        },
        {
            "foo": "bb7b2cd0-f140-4ffe-8280-dc462784303d",
            "action": "zebra",
        }

    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("foo", StringType()),
            StructField('action', StringType()),
        ])
    )

    result = input_df.transform(f)
    result_count = result.count()
    assert result_count == 1, f"expected 1, but got {result_count}"

    result_actions = [x.action for x in result.collect()]
    expected_actions = ["StopTransaction"]
    assert result_actions == expected_actions, f"expect {expected_actions}, but got {result_actions}"

    print("All tests pass! :)")


def test_return_stoptransaction_e2e(input_df: DataFrame, **kwargs):
    result = input_df

    count = result.count()
    expected_count = 95
    assert count == expected_count, f"expected {expected_count} got {count}"

    unique_actions = set([x["action"] for x in result.select("action").collect()])
    expected_actions = set(["StopTransaction"])
    assert unique_actions == expected_actions, f"expected {expected_actions}, but got {unique_actions}"

    print("All tests pass! :)")


def test_convert_stop_transaction_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "foo": "30e2ed0c-dd61-4fc1-bcb8-f0a8a0f87c0a",
            "body": json.dumps({
                "meter_stop": 26795,
                "timestamp": "2022-10-02T15:56:17.000345+00:00",
                "transaction_id": 1,
                "reason": None,
                "id_tag": "14902753768387952483",
                "transaction_data": None
            })
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("foo", StringType()),
            StructField("body", StringType()),
        ])
    )

    result = input_df.transform(f)

    print("Transformed DF:")
    result.show()

    result_count = result.count()
    assert result_count == 1

    def get_json_value(df: DataFrame, column: str, key: str):
        return [getattr(x, key) for x in df.select(col(f"{column}.{key}")).collect()][0]

    assert get_json_value(result, "new_body",
                          "meter_stop") == 26795, f"expected 26795, but got {get_json_value(result, 'new_body', 'meter_stop')}"
    assert get_json_value(result, "new_body",
                          "timestamp") == "2022-10-02T15:56:17.000345+00:00", f"expected '2022-10-02T15:56:17.000345+00:00', but got {get_json_value(result, 'new_body', 'timestamp')}"
    assert get_json_value(result, "new_body",
                          "transaction_id") == 1, f"expected 1, but got {get_json_value(result, 'new_body', 'transaction_id')}"
    assert get_json_value(result, "new_body",
                          "reason") == None, f"expected None, but got {get_json_value(result, 'new_body', 'reason')}"
    assert get_json_value(result, "new_body",
                          "id_tag") == "14902753768387952483", f"expected '14902753768387952483', but got {get_json_value(result, 'new_body', 'id_tag')}"
    assert get_json_value(result, "new_body",
                          "transaction_data") == None, f"expected None, but got {get_json_value(result, 'new_body', 'transaction_data')}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('foo', StringType(), True),
        StructField('body', StringType(), True),
        StructField('new_body',
                    StructType([
                        StructField('meter_stop', IntegerType(), True),
                        StructField('timestamp', StringType(), True),
                        StructField('transaction_id', IntegerType(), True),
                        StructField('reason', StringType(), True),
                        StructField('id_tag', StringType(), True),
                        StructField('transaction_data', ArrayType(StringType(), True), True)]),
                    True)
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")


def test_convert_stop_transaction_json_e2e(input_df, display_f: Callable, **kwargs):
    result = input_df

    print("Transformed DF:")
    result.show()

    assert result.columns == ["message_id", "message_type", "charge_point_id", "action", "write_timestamp", "body",
                              "new_body"]
    assert result.count() == 95, f"expected 95, but got {result.count()}"

    result_sub = result.withColumn("converted_timestamp", to_timestamp("new_body.timestamp")).sort(col("converted_timestamp")).drop("converted_timestamp").limit(3)
    print("Reordered DF under test:")
    display_f(result_sub)

    meter_stop = [x.meter_stop for x in result_sub.select(col("new_body.meter_stop")).collect()]
    expected_meter_stop = [51219, 31374, 50781]
    assert meter_stop == expected_meter_stop, f"expected {expected_meter_stop}, but got {meter_stop}"

    timestamps = [x.timestamp for x in result_sub.select(col("new_body.timestamp")).collect()]
    expected_timestamps = ['2023-01-01T17:11:31.399112+00:00', '2023-01-01T17:48:30.073819+00:00',
                           '2023-01-01T20:57:10.917742+00:00']
    assert timestamps == expected_timestamps, f"expected {expected_timestamps}, but got {timestamps}"

    transaction_ids = [x.transaction_id for x in result_sub.select(col("new_body.transaction_id")).collect()]
    expected_transaction_ids = [1, 5, 7]
    assert transaction_ids == expected_transaction_ids, f"expected {expected_transaction_ids}, but got {transaction_ids}"

    reasons = [x.reason for x in result_sub.select(col("new_body.reason")).collect()]
    expected_reasons = [None, None, None]
    assert reasons == expected_reasons, f"expected {expected_reasons}, but got {reasons}"

    id_tags = [x.id_tag for x in result_sub.select(col("new_body.id_tag")).collect()]
    expected_id_tags = ['e812abe5-e73b-453d-b71d-29ef6e1593f5', 'e812abe5-e73b-453d-b71d-29ef6e1593f5',
                        'e812abe5-e73b-453d-b71d-29ef6e1593f5']
    assert id_tags == expected_id_tags, f"expected {expected_id_tags}, but got {id_tags}"

    transaction_data = [x.transaction_data for x in result_sub.select(col("new_body.transaction_data")).collect()]
    expected_transaction_data = [None, None, None]
    assert transaction_data == expected_transaction_data, f"expected {expected_transaction_data}, but got {transaction_data}"

    print("All tests pass! :)")

def test_convert_start_transaction_request_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "foo": "30e2ed0c-dd61-4fc1-bcb8-f0a8a0f87c0a",
            "body": json.dumps({
                "connector_id": 1,
                "id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
                "meter_start": 0,
                "timestamp": "2022-01-01T08:00:00+00:00",
                "reservation_id": None
            })
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("foo", StringType()),
            StructField("body", StringType()),
        ])
    )

    result = input_df.transform(f)

    print("Transformed DF:")
    result.show()

    result_count = result.count()
    assert result_count == 1

    def get_json_value(df: DataFrame, column: str, key: str):
        return [getattr(x, key) for x in df.select(col(f"{column}.{key}")).collect()][0]

    assert get_json_value(result, "new_body",
                          "connector_id") == 1, f"expected 0, but got {get_json_value(result, 'new_body', 'connector_id')}"
    assert get_json_value(result, "new_body",
                          "id_tag") == "ea068c10-1bfb-4128-ab88-de565bd5f02f", f"expected 'ea068c10-1bfb-4128-ab88-de565bd5f02f', but got {get_json_value(result, 'new_body', 'id_tag')}"
    assert get_json_value(result, "new_body",
                          "meter_start") == 0, f"expected 0, but got {get_json_value(result, 'new_body', 'meter_start')}"
    assert get_json_value(result, "new_body",
                          "timestamp") == "2022-01-01T08:00:00+00:00", f"expected '2022-01-01T08:00:00+00:00', but got {get_json_value(result, 'new_body', 'timestamp')}"
    assert get_json_value(result, "new_body",
                          "reservation_id") == None, f"expected 1, but got {get_json_value(result, 'new_body', 'reservation_id')}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('foo', StringType(), True),
        StructField('body', StringType(), True),
        StructField('new_body',
                    StructType([
                        StructField('connector_id', IntegerType(), True),
                        StructField('id_tag', StringType(), True),
                        StructField('meter_start', IntegerType(), True),
                        StructField('timestamp', StringType(), True),
                        StructField('reservation_id', IntegerType(), True)]),
                    True)
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")


def test_convert_start_transaction_response_json_e2e(input_df: DataFrame, display_f: Callable, **kwargs):
    result = input_df

    print("Transformed DF:")
    display_f(result)

    assert result.columns == ["message_id", "message_type", "charge_point_id", "action", "write_timestamp", "body",
                              "new_body"]
    assert result.count() == 95, f"expected 95, but got {result.count()}"

    result_sub = result.sort(col("new_body.transaction_id")).limit(3)
    print("Reordered DF under test:")
    display_f(result_sub)

    def assert_expected_json_value(json_path: str, expected_values: List[Any]):
        values = [getattr(x, json_path.split(".")[-1]) for x in result_sub.select(col(json_path)).collect()]
        assert values == expected_values, f"expected {expected_values}, but got {values}"

    assert_expected_json_value("new_body.transaction_id", [1, 2, 3])
    assert_expected_json_value("new_body.id_tag_info.status", ['Accepted', 'Accepted', 'Accepted'])
    assert_expected_json_value("new_body.id_tag_info.parent_id_tag",
                               ['e812abe5-e73b-453d-b71d-29ef6e1593f5', 'e812abe5-e73b-453d-b71d-29ef6e1593f5',
                                'e812abe5-e73b-453d-b71d-29ef6e1593f5'])
    assert_expected_json_value("new_body.id_tag_info.expiry_date", [None, None, None])

    print("All tests pass! :)")


def test_convert_start_transaction_response_json_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "foo": "30e2ed0c-dd61-4fc1-bcb8-f0a8a0f87c0a",
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

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("foo", StringType()),
            StructField("body", StringType()),
        ])
    )

    result = input_df.transform(f)

    print("Transformed DF:")
    result.show()

    result_count = result.count()
    assert result_count == 1

    def get_json_value(df: DataFrame, column: str, key: str):
        return [getattr(x, key) for x in df.select(col(f"{column}.{key}")).collect()][0]

    assert get_json_value(result, "new_body",
                          "transaction_id") == 1, f"expected 1, but got {get_json_value(result, 'new_body', 'transaction_id')}"
    assert get_json_value(result, "new_body",
                          "id_tag_info") == Row(status='Accepted',
                                                parent_id_tag='ea068c10-1bfb-4128-ab88-de565bd5f02f',
                                                expiry_date=None), f"expected None, but got {get_json_value(result, 'new_body', 'id_tag_info')}"

    result_schema = result.schema
    id_tag_info_schema = StructType([
        StructField('status', StringType(), True),
        StructField('parent_id_tag', StringType(), True),
        StructField('expiry_date', StringType(), True),
    ])
    expected_schema = StructType([
        StructField('foo', StringType(), True),
        StructField('body', StringType(), True),
        StructField('new_body',
                    StructType([
                        StructField('transaction_id', IntegerType(), True),
                        StructField('id_tag_info', id_tag_info_schema, True),
                    ]),
                    True)
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")


def test_join_with_start_transaction_request_unit(spark, f: Callable):
    input_start_transaction_response = pd.DataFrame([
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

    input_start_transaction_response_df = spark.createDataFrame(
        input_start_transaction_response,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("message_id", StringType()),
            StructField("body", StringType()),
        ])
    )

    input_start_transaction_response_body_id_tag_info_schema = StructType([
        StructField("status", StringType(), True),
        StructField("parent_id_tag", StringType(), True),
        StructField("expiry_date", StringType(), True),
    ])

    input_start_transaction_response_body_schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("id_tag_info", input_start_transaction_response_body_id_tag_info_schema, True)
    ])

    input_start_transaction_response_converted_df = input_start_transaction_response_df.withColumn("new_body",from_json(col("body"), input_start_transaction_response_body_schema))


    input_start_transaction_request_pandas = pd.DataFrame([
        {
            "charge_point_id": "123",
            "message_id": "456",
            "body": json.dumps({
                "connector_id": 1,
                "id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
                "meter_start": 0,
                "timestamp": "2022-01-01T08:00:00+00:00",
                "reservation_id": None
            })
        },
    ])

    input_start_transaction_request_df = spark.createDataFrame(
        input_start_transaction_request_pandas,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("message_id", StringType()),
            StructField("body", StringType()),
        ])
    )

    input_start_transaction_request_body_schema = StructType([
        StructField("connector_id", IntegerType(), True),
        StructField("id_tag", StringType(), True),
        StructField("meter_start", IntegerType(), True),
        StructField("timestamp", StringType(), True),
        StructField("reservation_id", IntegerType(), True),
    ])

    input_start_transaction_request_converted_df = input_start_transaction_request_df.withColumn("new_body",from_json(col("body"), input_start_transaction_request_body_schema))

    result = input_start_transaction_response_converted_df.transform(f, input_start_transaction_request_converted_df)

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


def test_join_with_start_transaction_request_e2e(input_df: DataFrame, display_f: Callable, **kwargs):
    result = input_df

    print("Transformed DF:")
    display_f(result)

    assert result.columns == ["charge_point_id", "transaction_id", "meter_start", "start_timestamp"]
    assert result.count() == 95, f"expected 95, but got {result.count()}"

    result_sub = result.sort(col("transaction_id")).limit(3)
    print("Reordered DF under test:")
    display_f(result_sub)

    def assert_expected_value(column: str, expected_values: List[Any]):
        values = [getattr(x, column) for x in result_sub.select(col(column)).collect()]
        assert values == expected_values, f"expected {expected_values}, but got {values}"

    assert_expected_value("charge_point_id",
                          ['01a0f039-7685-4a7f-9ef6-8d262a7898fb', '3e365f3f-6e30-43d3-b897-d6291a9f7c35',
                           '77b7feb3-7f8f-4faf-86c6-d725e70e8c7f'])
    assert_expected_value("transaction_id", [1, 2, 3])
    assert_expected_value("meter_start", [0, 0, 0])
    assert_expected_value("start_timestamp", ['2023-01-01T12:54:04.750286+00:00', '2023-01-01T12:57:35.483812+00:00', '2023-01-01T13:48:12.471750+00:00'])

    print("All tests pass! :)")


def test_convert_start_transaction_request_json_e2e(input_df: DataFrame, display_f: Callable, **kwargs):
    result = input_df

    print("Transformed DF:")
    display_f(result)

    assert result.columns == ["message_id", "message_type", "charge_point_id", "action", "write_timestamp", "body",
                              "new_body"]
    assert result.count() == 95, f"expected 95, but got {result.count()}"

    result_sub = result.withColumn("converted_timestamp", to_timestamp("new_body.timestamp")).drop("converted_timestamp").limit(3)
    print("Reordered DF under test:")
    display_f(result_sub)

    def assert_expected_json_value(json_path: str, expected_values: List[Any]):
        values = [getattr(x, json_path.split(".")[-1]) for x in result_sub.select(col(json_path)).collect()]
        assert values == expected_values, f"expected {expected_values}, but got {values}"

    assert_expected_json_value("new_body.connector_id", [1, 2, 1])
    assert_expected_json_value("new_body.id_tag",
                               ['e812abe5-e73b-453d-b71d-29ef6e1593f5', 'e812abe5-e73b-453d-b71d-29ef6e1593f5',
                                'e812abe5-e73b-453d-b71d-29ef6e1593f5'])
    assert_expected_json_value("new_body.meter_start", [0, 0, 0])
    assert_expected_json_value("new_body.timestamp",
                               ['2023-01-01T12:54:04.750286+00:00', '2023-01-01T12:57:35.483812+00:00',
                                '2023-01-01T13:48:12.471750+00:00'])
    assert_expected_json_value("new_body.reservation_id", [None, None, None])

    print("All tests pass! :)")


def test_join_stop_with_start_unit(spark, f: Callable):
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
            "body": json.dumps({
                "meter_stop": 2780,
                "timestamp": "2022-01-01T08:20:00+00:00",
                "transaction_id": 1,
                "reason": None,
                "id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
                "transaction_data": None
            }),
        }
    ])

    input_stop_transaction_response_body_schema = StructType([
        StructField("meter_stop", IntegerType(), True),
        StructField("timestamp", StringType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("reason", StringType(), True),
        StructField("id_tag", StringType(), True),
        StructField("transaction_data", StringType(), True),
    ])

    input_stop_transaction_request_schema = StructType([
        StructField("foo", StringType(), True),
        StructField("body", StringType(), True)
    ])

    input_stop_transaction_request_df = spark.createDataFrame(
        input_stop_transaction_request_pandas,
        input_stop_transaction_request_schema
    )

    input_stop_transaction_request_converted_df = input_stop_transaction_request_df.withColumn("new_body",from_json(col("body"), input_stop_transaction_response_body_schema))

    result = input_stop_transaction_request_converted_df.transform(f, input_start_transaction_df)

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


def test_join_stop_with_start_e2e(input_df: DataFrame, display_f: Callable, **kwargs):
    result = input_df

    print("Transformed DF:")
    display_f(result)

    assert set(result.columns) == {"charge_point_id", "transaction_id", "meter_start", "meter_stop", "start_timestamp",
                                   "stop_timestamp"}
    assert result.count() == 95, f"expected 95, but got {result.count()}"

    result_sub = result.sort(col("transaction_id")).limit(3)
    print("Reordered DF under test:")
    display_f(result_sub)

    def assert_expected_value(column: str, expected_values: List[Any]):
        values = [getattr(x, column) for x in result_sub.select(col(column)).collect()]
        assert values == expected_values, f"expected {expected_values} in column {column}, but got {values}"

    assert_expected_value("charge_point_id",
                          ['01a0f039-7685-4a7f-9ef6-8d262a7898fb', '3e365f3f-6e30-43d3-b897-d6291a9f7c35', '77b7feb3-7f8f-4faf-86c6-d725e70e8c7f'])
    assert_expected_value("transaction_id", [1, 2, 3])
    assert_expected_value("meter_start", [0, 0, 0])
    assert_expected_value("meter_stop", [51219, 146616, 151794])
    assert_expected_value("start_timestamp", ['2023-01-01T12:54:04.750286+00:00', '2023-01-01T12:57:35.483812+00:00', '2023-01-01T13:48:12.471750+00:00'])
    assert_expected_value("stop_timestamp", ['2023-01-01T17:11:31.399112+00:00', '2023-01-01T22:21:00.811842+00:00', '2023-01-01T21:41:39.731897+00:00'])

    print("All tests pass! :)")


def test_calculate_total_time_hours_unit(spark, f: Callable):
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


def test_calculate_total_time_hours_e2e(input_df: DataFrame, **kwargs):
    result = input_df
    print("Transformed DF:")
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 95
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("charge_point_id", StringType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("meter_start", IntegerType(), True),
        StructField("meter_stop", IntegerType(), True),
        StructField("start_timestamp", TimestampType(), True),
        StructField("stop_timestamp", TimestampType(), True),
        StructField("total_time", DoubleType(), True)
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    result_total_time = [x.total_time for x in result.sort(col("transaction_id")).collect()]
    expected_total_time = [4.29, 9.39, 7.89, 7.5, 2.47, 10.84, 3.15, 2.68, 6.03, 2.03, 4.03, 4.18, 4.64, 2.7, 9.21,
                           7.05, 10.5, 8.55, 8.9, 11.95, 11.38, 10.25, 3.55, 3.82, 9.17, 6.19, 6.28, 11.35, 4.18, 11.92,
                           2.16, 7.88, 8.44, 4.75, 7.14, 6.52, 5.76, 11.11, 9.44, 8.61, 2.7, 5.2, 8.04, 3.19, 3.37,
                           11.94, 10.39, 10.9, 2.02, 2.56, 10.33, 6.94, 4.88, 7.81, 5.56, 4.21, 2.97, 11.87, 9.16, 3.24,
                           7.23, 6.97, 11.86, 6.41, 5.96, 7.4, 9.02, 10.28, 4.87, 5.46, 10.53, 7.68, 10.93, 6.84, 7.09,
                           4.94, 10.84, 5.81, 5.36, 8.9, 5.56, 9.05, 2.48, 2.58, 2.91, 8.91, 8.87, 3.51, 10.82, 7.03,
                           8.92, 5.93, 2.03, 2.96, 2.28]
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


def test_calculate_total_energy_e2e(input_df: DataFrame, **kwargs):
    result = input_df

    print("Transformed DF:")
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 95
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
    result_total_energy = [x.total_energy for x in result_ordered.collect()]
    expected_total_energy = [51219.0, 146616.0, 151794.0, 106126.0, 31374.0, 193968.0, 50781.0, 42121.0, 95634.0,
                             23897.0, 43316.0, 43746.0, 77118.0, 34277.0, 144768.0, 98641.0, 170171.0, 137738.0,
                             149056.0, 199170.0, 227549.0, 117548.0, 42235.0, 48498.0, 145084.0, 83495.0, 76078.0,
                             174636.0, 74102.0, 177470.0, 25978.0, 144815.0, 105303.0, 86140.0, 133118.0, 102056.0,
                             92845.0, 176318.0, 136581.0, 155487.0, 36414.0, 96265.0, 125985.0, 37903.0, 52334.0,
                             211115.0, 182410.0, 157962.0, 21851.0, 23476.0, 164136.0, 95713.0, 86874.0, 104892.0,
                             75476.0, 60495.0, 47719.0, 229061.0, 128245.0, 43527.0, 94194.0, 112741.0, 210995.0,
                             98534.0, 98066.0, 116117.0, 147795.0, 147573.0, 62259.0, 73185.0, 197632.0, 127848.0,
                             172165.0, 74999.0, 105432.0, 78858.0, 198323.0, 101860.0, 73797.0, 145058.0, 83244.0,
                             151649.0, 29350.0, 33778.0, 38108.0, 123547.0, 149542.0, 37542.0, 160941.0, 95735.0,
                             158472.0, 91462.0, 25614.0, 29244.0, 25278.0]
    assert result_total_energy == expected_total_energy, f"expected {expected_total_energy}, but got {result_total_energy}"
    print("All tests passed! :)")


def test_convert_start_stop_timestamp_to_timestamp_type_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "transaction_id": 1,
            "charge_point_id": '123',
            "meter_start": 0,
            "meter_stop": 1000,
            "start_timestamp": '2022-10-01T13:23:34.000235+00:00',
            "stop_timestamp": '2022-10-02T15:56:17.000345+00:00',
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("transaction_id", IntegerType(), True),
            StructField("charge_point_id", StringType(), True),
            StructField("meter_start", IntegerType(), True),
            StructField("meter_stop", IntegerType(), True),
            StructField("start_timestamp", StringType(), True),
            StructField("stop_timestamp", StringType(), True),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF:")
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_columns = result.columns
    expected_columns = ["transaction_id", "charge_point_id", "meter_start", "meter_stop",
                        "start_timestamp", "stop_timestamp"]
    assert result_columns == expected_columns, f"expected {expected_columns}, but got {result_columns}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("charge_point_id", StringType(), True),
        StructField("meter_start", IntegerType(), True),
        StructField("meter_stop", IntegerType(), True),
        StructField("start_timestamp", TimestampType(), True),
        StructField("stop_timestamp", TimestampType(), True),
    ])
    assert result_schema == expected_schema

    print("All tests pass! :)")


def test_convert_start_stop_timestamp_to_timestamp_type_e2e(input_df: DataFrame, display_f: Callable, **kwargs):
    result = input_df

    print("Transformed DF:")
    display_f(result)

    result_count = result.count()
    expected_count = 95
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("charge_point_id", StringType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("meter_start", IntegerType(), True),
        StructField("meter_stop", IntegerType(), True),
        StructField("start_timestamp", TimestampType(), True),
        StructField("stop_timestamp", TimestampType(), True)
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    print("All tests passed! :)")


def test_convert_metervalues_to_json_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": "AL1000",
            "write_timestamp": "2022-10-02T15:30:17.000345+00:00",
            "action": "MeterValues",
            "body": '{"connector_id": 1, "meter_value": [{"timestamp": "2022-10-02T15:30:17.000345+00:00", "sampled_value": [{"value": "0.00", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L1-N", "location": "Outlet", "unit": "V"}, {"value": "13.17", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L1", "location": "Outlet", "unit": "A"}, {"value": "3663.49", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L1", "location": "Outlet", "unit": "W"}, {"value": "238.65", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L2-N", "location": "Outlet", "unit": "V"}, {"value": "14.28", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L2", "location": "Outlet", "unit": "A"}, {"value": "3086.46", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L2", "location": "Outlet", "unit": "W"}, {"value": "215.21", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L3-N", "location": "Outlet", "unit": "V"}, {"value": "14.63", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L3", "location": "Outlet", "unit": "A"}, {"value": "4014.47", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L3", "location": "Outlet", "unit": "W"}, {"value": "254.65", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": null, "location": "Outlet", "unit": "Wh"}, {"value": "11.68", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L1-N", "location": "Outlet", "unit": "V"}, {"value": "3340.61", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L1", "location": "Outlet", "unit": "A"}, {"value": "7719.95", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L1", "location": "Outlet", "unit": "W"}, {"value": "0.00", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L2-N", "location": "Outlet", "unit": "V"}, {"value": "3.72", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L2", "location": "Outlet", "unit": "A"}, {"value": "783.17", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L2", "location": "Outlet", "unit": "W"}, {"value": "242.41", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L3-N", "location": "Outlet", "unit": "V"}, {"value": "3.46", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L3", "location": "Outlet", "unit": "A"}, {"value": "931.52", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L3", "location": "Outlet", "unit": "W"}, {"value": "7.26", "context": "Sample.Periodic", "format": "Raw", "measurand": "Energy.Active.Import.Register", "phase": null, "location": "Outlet", "unit": "Wh"}]}], "transaction_id": 1}'
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("write_timestamp", StringType(), True),
            StructField("action", StringType(), True),
            StructField("body", StringType(), True),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF:")
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("charge_point_id", StringType(), True),
        StructField("write_timestamp", StringType(), True),
        StructField("action", StringType(), True),
        StructField("body", StringType(), True),
        StructField("new_body", StructType([
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
                    StructField("unit", StringType(), True)]), True), True)]), True), True)]), True)])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    print("All tests passed! :)")


def test_convert_metervalues_to_json_e2e(input_df: DataFrame, **kwargs):
    result = input_df
    print("Transformed DF:")
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 7767
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("message_id", StringType(), True),
        StructField("message_type", IntegerType(), True),
        StructField("charge_point_id", StringType(), True),
        StructField("action", StringType(), True),
        StructField("write_timestamp", StringType(), True),
        StructField("body", StringType(), True),
        StructField("new_body", StructType([
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
                    StructField("unit", StringType(), True)]), True), True)]), True), True)]), True)])

    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    print("All tests passed! :)")


def test_reshape_meter_values_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": "AL1000",
            "action": "MeterValues",
            "body": '{"connector_id": 1, "meter_value": [{"timestamp": "2022-10-02T15:30:17.000345+00:00", "sampled_value": [{"value": "0.00", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L1-N", "location": "Outlet", "unit": "V"}, {"value": "13.17", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L1", "location": "Outlet", "unit": "A"}, {"value": "3663.49", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L1", "location": "Outlet", "unit": "W"}, {"value": "238.65", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L2-N", "location": "Outlet", "unit": "V"}, {"value": "14.28", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L2", "location": "Outlet", "unit": "A"}, {"value": "3086.46", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L2", "location": "Outlet", "unit": "W"}, {"value": "215.21", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L3-N", "location": "Outlet", "unit": "V"}, {"value": "14.63", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L3", "location": "Outlet", "unit": "A"}, {"value": "4014.47", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L3", "location": "Outlet", "unit": "W"}, {"value": "254.65", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": null, "location": "Outlet", "unit": "Wh"}, {"value": "11.68", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L1-N", "location": "Outlet", "unit": "V"}, {"value": "3340.61", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L1", "location": "Outlet", "unit": "A"}, {"value": "7719.95", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L1", "location": "Outlet", "unit": "W"}, {"value": "0.00", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L2-N", "location": "Outlet", "unit": "V"}, {"value": "3.72", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L2", "location": "Outlet", "unit": "A"}, {"value": "783.17", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L2", "location": "Outlet", "unit": "W"}, {"value": "242.41", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L3-N", "location": "Outlet", "unit": "V"}, {"value": "3.46", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L3", "location": "Outlet", "unit": "A"}, {"value": "931.52", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L3", "location": "Outlet", "unit": "W"}, {"value": "1330", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": null, "location": "Outlet", "unit": "W"},{"value": "7.26", "context": "Sample.Periodic", "format": "Raw", "measurand": "Energy.Active.Import.Register", "phase": null, "location": "Outlet", "unit": "Wh"}]}], "transaction_id": 1}'
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("action", StringType(), True),
            StructField("body", StringType(), True),
        ])
    )

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
                    StructField("unit", StringType(), True)]), True), True)]), True), True)])

    input_df = input_df.withColumn("new_body", from_json(col("body"), body_schema))
    result = input_df.transform(f)
    print("Transformed DF:")
    result.show()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_columns = set(result.columns)
    expected_columns = {"transaction_id", "timestamp", "measurand", "phase", "value"}
    assert result_columns == expected_columns, f"Expected {expected_columns}, but got {result_columns}"

    result_value = [x.value for x in result.collect()]
    expected_value = [1330]
    assert result_value == expected_value, f"Expected {expected_value}, but got {result_value}"

    print("All tests pass! :)")


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
    expected_count = 94
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('transaction_id', IntegerType(), True),
        StructField('total_parking_time', DoubleType(), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_total_parking_time = [x.total_parking_time for x in result.collect()]
    expected_total_parking_time = [1.0, 1.75, 0.25, 1.0, 0.67, 1.25, 0.42, 0.33, 0.75, 0.58, 1.42, 1.42, 0.75, 0.75, 0.92, 0.92, 2.0, 1.33, 0.83, 1.92, 0.5, 2.83, 0.83, 1.08, 0.17, 1.67, 1.75, 1.33, 0.17, 0.75, 0.5, 0.92, 1.92, 0.42, 0.67, 0.5, 0.25, 1.67, 2.0, 0.83, 0.42, 0.58, 1.08, 1.25, 0.25, 1.33, 1.5, 2.0, 0.5, 1.0, 0.75, 1.25, 1.92, 0.75, 0.67, 0.42, 0.5, 1.17, 0.42, 2.25, 1.0, 1.58, 0.17, 0.75, 0.5, 1.08, 1.58, 1.33, 1.08, 0.25, 0.67, 1.33, 2.42, 1.08, 0.92, 1.42, 0.17, 1.58, 0.5, 1.33, 0.83, 0.83, 0.67, 0.75, 1.08, 1.25, 1.0, 1.0, 2.0, 0.92, 0.58, 0.42, 1.17, 0.75]
    assert result_total_parking_time == expected_total_parking_time, f"Expected {expected_total_parking_time}, but got {result_total_parking_time}"

    print("All tests pass! :)")


def test_join_with_target_df_unit(spark, f: Callable):
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


def test_join_with_target_df_e2e(input_df: DataFrame, display_f: Callable, **kwargs):
    result = input_df

    result_sub = result.sort(col("transaction_id"))
    print("Reordered DF under test:")
    display_f(result_sub)

    result_schema = result_sub.schema
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

    result_total_parking_time = [x.total_parking_time for x in result_sub.collect()]
    expected_total_parking_time = [1.0, 1.75, 0.25, 1.0, 0.67, 1.25, 0.42, 0.33, 0.75, 0.58, 1.42, 1.42, 0.75, 0.75, 0.92, 0.92, 2.0, 1.33, 0.83, 1.92, 0.5, 2.83, 0.83, 1.08, 0.17, 1.67, 1.75, 1.33, 0.17, 0.75, 0.5, 0.92, 1.92, 0.42, 0.67, 0.5, 0.25, 1.67, 2.0, 0.83, 0.42, 0.58, 1.08, 1.25, 0.25, 1.33, 1.5, 2.0, 0.5, 1.0, 0.75, 1.25, None, 1.92, 0.75, 0.67, 0.42, 0.5, 1.17, 0.42, 2.25, 1.0, 1.58, 0.17, 0.75, 0.5, 1.08, 1.58, 1.33, 1.08, 0.25, 0.67, 1.33, 2.42, 1.08, 0.92, 1.42, 0.17, 1.58, 0.5, 1.33, 0.83, 0.83, 0.67, 0.75, 1.08, 1.25, 1.0, 1.0, 2.0, 0.92, 0.58, 0.42, 1.17, 0.75]
    assert result_total_parking_time == expected_total_parking_time, f"Expected {expected_total_parking_time}, but got {result_total_parking_time}"

    print("All tests pass!")


def test_cleanup_columns_unit(spark, f: Callable):
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
            "total_parking_time": 0.1
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
            "total_parking_time": None
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
            "total_parking_time": 0.2
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
            StructField("total_parking_time", DoubleType(), True),
        ])
    )


    result = input_df.transform(f)
    print("Transformed DF:")
    result.show()

    result_count = result.count()
    expected_count = 3
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("charge_point_id", StringType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("start_timestamp", TimestampType(), True),
        StructField("stop_timestamp", TimestampType(), True),
        StructField("total_time", DoubleType(), True),
        StructField("total_energy", DoubleType(), True),
        StructField("total_parking_time", DoubleType(), True),
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")



def test_cleanup_columns_e2e(input_df: DataFrame, display_f: Callable, **kwargs):
    result = input_df
    display_f(result)

    result_count = result.count()
    expected_count = 95
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("charge_point_id", StringType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("start_timestamp", TimestampType(), True),
        StructField("stop_timestamp", TimestampType(), True),
        StructField("total_time", DoubleType(), True),
        StructField("total_energy", DoubleType(), True),
        StructField("total_parking_time", DoubleType(), True),
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    print("All tests pass!")

